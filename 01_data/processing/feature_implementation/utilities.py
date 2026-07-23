"""Shared helpers for feature-implementation modules."""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------


def log_return(close: pd.Series) -> pd.Series:
    """``ln(C_t / C_{t-1})``; first bar is NaN."""
    c = close.astype(float)
    return np.log(c / c.shift(1))


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
