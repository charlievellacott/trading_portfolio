"""H-010 short-selling pressure: series primitives and panel helpers."""

from __future__ import annotations

import pandas as pd

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
)


# ---------------------------------------------------------------------------
# Series primitives
# ---------------------------------------------------------------------------


def short_volume_ratio(
    short_volume: pd.Series,
    total_volume: pd.Series,
) -> pd.Series:
    """``short_volume / total_volume``; NaN when ``total_volume <= 0``."""
    short = short_volume.astype(float)
    total = total_volume.astype(float)
    return (short / total).where(total > 0)


def abnormal_short_flow(
    svr: pd.Series,
    *,
    smooth_window: int = 5,
    baseline_window: int = 60,
) -> pd.Series:
    """
    Z-score of smoothed short-volume ratio vs own-history baseline.

    ``sm = svr.rolling(smooth_window).mean()``
    ``base = sm.shift(1).rolling(baseline_window)`` (excludes current obs)
    ``z = (sm - base.mean()) / base.std(ddof=0)``; NaN when ``std == 0``.
    """
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")
    if baseline_window < 1:
        raise ValueError("baseline_window must be >= 1")

    sm = svr.astype(float).rolling(smooth_window, min_periods=smooth_window).mean()
    lagged = sm.shift(1)
    base_mean = lagged.rolling(baseline_window, min_periods=baseline_window).mean()
    base_std = lagged.rolling(baseline_window, min_periods=baseline_window).std(ddof=0)
    z = (sm - base_mean) / base_std
    return z.where(base_std > 0)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def add_short_volume_ratio(
    panel: pd.DataFrame,
    *,
    col: str = "short_volume_ratio",
    short_col: str = "short_volume",
    total_col: str = "total_volume",
) -> pd.DataFrame:
    """Return a copy with per-row short-volume ratio in ``col``."""
    _require_columns(panel, {"date", "ticker", short_col, total_col})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    result = panel.copy()
    result[col] = short_volume_ratio(result[short_col], result[total_col])
    return result


def add_abnormal_short_flow(
    panel: pd.DataFrame,
    *,
    smooth_window: int = 5,
    baseline_window: int = 60,
    col: str = "abnormal_short_flow",
    svr_col: str = "short_volume_ratio",
) -> pd.DataFrame:
    """
    Return a copy with per-ticker abnormal short flow in ``col``.

    Requires ``svr_col`` (typically from ``add_short_volume_ratio``). Baseline
    rolling stats exclude the current observation via ``shift(1)``.
    """
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")
    if baseline_window < 1:
        raise ValueError("baseline_window must be >= 1")
    _require_columns(panel, {"date", "ticker", svr_col})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = work.groupby("ticker", sort=False)[svr_col].transform(
        lambda s: abnormal_short_flow(
            s,
            smooth_window=smooth_window,
            baseline_window=baseline_window,
        )
    )
    return _restore_order(work, original_index)
