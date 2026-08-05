"""Shared helpers for feature-implementation modules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _require_columns(panel: pd.DataFrame, required: set[str] | frozenset[str]) -> None:
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")


def _sorted_by_ticker_date(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(["ticker", "date"], kind="mergesort")


def _restore_order(result: pd.DataFrame, original_index: pd.Index) -> pd.DataFrame:
    return result.reindex(original_index)


def merge_info_dated(
    panel: pd.DataFrame,
    other: pd.DataFrame,
    cols: list[str] | tuple[str, ...],
    *,
    how: str = "left",
) -> pd.DataFrame:
    """
    Left-merge ``other`` onto ``panel`` on the information cutoff date.

    S1 trade-date panels carry ``feature_date`` (previous bar). Market returns
    and other calendar-dated series must join on ``feature_date``, not trade
    ``date``. Close-dated panels (no ``feature_date``) join on ``date``.
    """
    if "date" not in other.columns:
        raise ValueError("other missing column: 'date'")
    use_cols = ["date", *cols]
    missing = set(use_cols) - set(other.columns)
    if missing:
        raise ValueError(f"other missing columns: {sorted(missing)}")
    right = other[use_cols]
    if "feature_date" in panel.columns:
        right = right.rename(columns={"date": "feature_date"})
        return panel.merge(right, on="feature_date", how=how)
    return panel.merge(right, on="date", how=how)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def cross_sectional_pct_rank(
    panel: pd.DataFrame,
    col: str,
    *,
    by: str = "date",
) -> pd.Series:
    """Percentile rank of ``col`` within each ``by`` group, values in [0, 1]."""
    if col not in panel.columns:
        raise ValueError(f"panel missing column: {col!r}")
    if by not in panel.columns:
        raise ValueError(f"panel missing column: {by!r}")
    return panel.groupby(by, sort=False)[col].rank(pct=True, method="average")


def cross_sectional_zscore(
    panel: pd.DataFrame,
    col: str,
    *,
    by: str = "date",
) -> pd.Series:
    """
    Z-score of ``col`` within each ``by`` group: ``(x - mean) / std``.

    Uses sample std (``ddof=1``). Returns NaN when the group std is 0 or when
    ``x`` is non-finite.
    """
    if col not in panel.columns:
        raise ValueError(f"panel missing column: {col!r}")
    if by not in panel.columns:
        raise ValueError(f"panel missing column: {by!r}")

    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=1)
        if std is None or not np.isfinite(std) or std == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - s.mean()) / std

    return panel.groupby(by, sort=False)[col].transform(_z)


def cross_sectional_ols_residual(
    panel: pd.DataFrame,
    y_col: str,
    x_col: str,
    *,
    by: str = "date",
) -> pd.Series:
    """
    Within each ``by`` group, OLS residual of ``y_col ~ 1 + x_col``.

    Groups with fewer than 3 finite ``(y, x)`` pairs or a singular design
    yield NaN for all rows in that group. Output is aligned to ``panel.index``.
    """
    if y_col not in panel.columns:
        raise ValueError(f"panel missing column: {y_col!r}")
    if x_col not in panel.columns:
        raise ValueError(f"panel missing column: {x_col!r}")
    if by not in panel.columns:
        raise ValueError(f"panel missing column: {by!r}")

    out = pd.Series(np.nan, index=panel.index, dtype=float)

    def _group_resid(grp: pd.DataFrame) -> pd.Series:
        y = grp[y_col].astype(float)
        x = grp[x_col].astype(float)
        ok = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        resid = pd.Series(np.nan, index=grp.index, dtype=float)
        if int(ok.sum()) < 3:
            return resid
        yv = y.loc[ok].to_numpy(dtype=float)
        xv = x.loc[ok].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(xv)), xv])
        try:
            coef, _, rank, _ = np.linalg.lstsq(design, yv, rcond=None)
        except np.linalg.LinAlgError:
            return resid
        if rank < 2:
            return resid
        resid.loc[ok] = yv - design @ coef
        return resid

    for _, grp in panel.groupby(by, sort=False):
        out.loc[grp.index] = _group_resid(grp).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------


def daily_simple_return(close: pd.Series) -> pd.Series:
    """``C_t / C_{t-1} - 1``; non-positive / non-finite closes → NaN."""
    c = close.astype(float)
    prev = c.shift(1)
    ok = (
        c.notna()
        & prev.notna()
        & np.isfinite(c)
        & np.isfinite(prev)
        & (c > 0)
        & (prev > 0)
    )
    return (c / prev - 1.0).where(ok)


def log_return(close: pd.Series) -> pd.Series:
    """``ln(C_t / C_{t-1})``; first bar / non-positive closes → NaN."""
    c = close.astype(float)
    prev = c.shift(1)
    ok = (
        c.notna()
        & prev.notna()
        & np.isfinite(c)
        & np.isfinite(prev)
        & (c > 0)
        & (prev > 0)
    )
    return np.log(c / prev).where(ok)


def abnormal_z(
    series: pd.Series,
    *,
    smooth_window: int = 5,
    baseline_window: int = 60,
) -> pd.Series:
    """
    Own-history z-score of a smoothed series (H-010-style).

    ``sm = series.rolling(smooth_window).mean()``
    ``base = sm.shift(1).rolling(baseline_window)``
    ``z = (sm - base.mean()) / base.std(ddof=0)``; NaN when ``std == 0``.
    """
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")
    if baseline_window < 1:
        raise ValueError("baseline_window must be >= 1")
    sm = series.astype(float).rolling(smooth_window, min_periods=smooth_window).mean()
    lagged = sm.shift(1)
    base_mean = lagged.rolling(baseline_window, min_periods=baseline_window).mean()
    base_std = lagged.rolling(baseline_window, min_periods=baseline_window).std(ddof=0)
    z = (sm - base_mean) / base_std
    return z.where(base_std > 0)


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------


def normalize_windows(windows: int | list[int] | tuple[int, ...]) -> list[int]:
    """Normalize ``windows`` to a non-empty list of positive ints."""
    if isinstance(windows, bool):
        raise ValueError("windows must be a positive int or a list of positive ints")
    if isinstance(windows, int):
        items = [windows]
    elif isinstance(windows, (list, tuple)):
        items = list(windows)
    else:
        raise ValueError("windows must be a positive int or a list of positive ints")
    if not items:
        raise ValueError("windows must be a non-empty list of positive ints")
    for w in items:
        if not isinstance(w, int) or isinstance(w, bool) or w < 1:
            raise ValueError(f"windows entries must be positive ints, got {w!r}")
    return items


def resolve_feature_subset(
    feature_subset: Sequence[str] | None,
    available: tuple[str, ...],
    *,
    name: str,
) -> list[str]:
    """
    Resolve ``feature_subset`` against a canonical ``available`` ID tuple.

    ``None`` or empty → all ``available`` IDs in that order.
    Non-empty → dedupe preserving caller order; unknown ID → ``ValueError``.
    """
    if feature_subset is None or len(feature_subset) == 0:
        return list(available)
    available_set = set(available)
    out: list[str] = []
    seen: set[str] = set()
    for item in feature_subset:
        if item not in available_set:
            raise ValueError(
                f"{name}: unknown feature_subset id {item!r}; "
                f"expected one of {list(available)}"
            )
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def regression_column_name(metric: str, window: int, *, multi_window: bool) -> str:
    """Bare ``metric`` when one window; ``metric_{window}`` when multiple."""
    if multi_window:
        return f"{metric}_{window}"
    return metric


def windowed_column_name(stem: str, *parts: int, multi: bool) -> str:
    """Bare ``stem`` when one combo; ``stem_{p0}_{p1}_...`` when ``multi``."""
    if not multi:
        return stem
    if not parts:
        raise ValueError("parts must be non-empty when multi=True")
    return stem + "".join(f"_{p}" for p in parts)
