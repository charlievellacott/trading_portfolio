"""H-005 Size & Value: series primitives and panel helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.linear_regression import (
    rolling_ols_stats,
)
from data.processing.feature_implementation.momentum import raw_momentum
from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    cross_sectional_pct_rank,
)

_REQUIRED_SV = frozenset({"date", "ticker", "close", "market_cap", "pe", "pb"})
_VALID_METRICS = frozenset({"pe", "pb"})


# ---------------------------------------------------------------------------
# Series primitives
# ---------------------------------------------------------------------------


def book_yield(pb: pd.Series) -> pd.Series:
    """``1 / pb``; NaN when ``pb <= 0``."""
    p = pb.astype(float)
    return (1.0 / p).where(p > 0)


def earnings_yield(pe: pd.Series) -> pd.Series:
    """``1 / pe``; NaN when ``pe <= 0``."""
    p = pe.astype(float)
    return (1.0 / p).where(p > 0)


def log_market_cap(mcap: pd.Series) -> pd.Series:
    """``log(market_cap)``; NaN when ``mcap <= 0``."""
    m = mcap.astype(float)
    return np.log(m).where(m > 0)


def valuation_roc(val: pd.Series, window: int) -> pd.Series:
    """``log(val_t) - log(val_{t-L})``; NaN when ``val <= 0``."""
    if window < 1:
        raise ValueError("window must be >= 1")
    v = val.astype(float)
    log_v = np.log(v).where(v > 0)
    return log_v - log_v.shift(window)


def size_momentum(mcap: pd.Series, window: int) -> pd.Series:
    """``log(mcap_t / mcap_{t-L})``; NaN when ``mcap <= 0``."""
    if window < 1:
        raise ValueError("window must be >= 1")
    m = mcap.astype(float)
    log_m = np.log(m).where(m > 0)
    return log_m - log_m.shift(window)


def value_momentum_distance(val_rank: pd.Series, mom_rank: pd.Series) -> pd.Series:
    """``sqrt((1 - mom_rank)**2 + (1 - val_rank)**2)``."""
    vr = val_rank.astype(float)
    mr = mom_rank.astype(float)
    return np.sqrt((1.0 - mr) ** 2 + (1.0 - vr) ** 2)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def add_book_yield(
    panel: pd.DataFrame,
    *,
    col: str = "book_yield",
) -> pd.DataFrame:
    """Return a copy with per-row book yield (``1/pb``) in ``col``."""
    _require_columns(panel, {"date", "ticker", "pb"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    result = panel.copy()
    result[col] = book_yield(result["pb"])
    return result


def add_earnings_yield(
    panel: pd.DataFrame,
    *,
    col: str = "earnings_yield",
) -> pd.DataFrame:
    """Return a copy with per-row earnings yield (``1/pe``) in ``col``."""
    _require_columns(panel, {"date", "ticker", "pe"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    result = panel.copy()
    result[col] = earnings_yield(result["pe"])
    return result


def add_log_market_cap(
    panel: pd.DataFrame,
    *,
    col: str = "log_mcap",
) -> pd.DataFrame:
    """Return a copy with ``log(market_cap)`` in ``col``."""
    _require_columns(panel, {"date", "ticker", "market_cap"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    result = panel.copy()
    result[col] = log_market_cap(result["market_cap"])
    return result


def add_valuation_roc(
    panel: pd.DataFrame,
    *,
    metric: str = "pb",
    window: int = 63,
    col: str = "val_roc",
) -> pd.DataFrame:
    """
    Return a copy with valuation rate-of-change in ``col``.

    ``metric`` selects which column (``'pe'`` or ``'pb'``).
    """
    if metric not in _VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(_VALID_METRICS)}, got {metric!r}")
    if window < 1:
        raise ValueError("window must be >= 1")
    _require_columns(panel, {"date", "ticker", metric})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = work.groupby("ticker", sort=False)[metric].transform(
        lambda s: valuation_roc(s, window)
    )
    return _restore_order(work, original_index)


def add_size_momentum(
    panel: pd.DataFrame,
    *,
    window: int = 63,
    col: str = "size_mom",
) -> pd.DataFrame:
    """Return a copy with ``log(mcap_t / mcap_{t-L})`` in ``col``."""
    if window < 1:
        raise ValueError("window must be >= 1")
    _require_columns(panel, {"date", "ticker", "market_cap"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = work.groupby("ticker", sort=False)["market_cap"].transform(
        lambda s: size_momentum(s, window)
    )
    return _restore_order(work, original_index)


def add_value_momentum_interaction(
    panel: pd.DataFrame,
    *,
    mom_lookback: int = 252,
    mom_skip: int = 21,
    col: str = "val_mom_interact",
) -> pd.DataFrame:
    """
    Return a copy with ``cs_rank(book_yield) * cs_rank(raw_momentum)`` in ``col``.

    Computes cross-sectional ranks on each date, then multiplies.
    """
    _require_columns(panel, {"date", "ticker", "close", "pb"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())

    work["_by"] = book_yield(work["pb"])
    work["_mom"] = work.groupby("ticker", sort=False)["close"].transform(
        lambda s: raw_momentum(s, lookback=mom_lookback, skip=mom_skip)
    )
    work["_by_rank"] = cross_sectional_pct_rank(work, "_by")
    work["_mom_rank"] = cross_sectional_pct_rank(work, "_mom")
    work[col] = work["_by_rank"] * work["_mom_rank"]

    work = work.drop(columns=["_by", "_mom", "_by_rank", "_mom_rank"])
    return _restore_order(work, original_index)


def add_value_momentum_distance(
    panel: pd.DataFrame,
    *,
    mom_lookback: int = 252,
    mom_skip: int = 21,
    col: str = "val_mom_dist",
) -> pd.DataFrame:
    """
    Return a copy with the value-momentum distance metric in ``col``.

    ``sqrt((1 - cs_rank(mom))^2 + (1 - cs_rank(book_yield))^2)``

    The ideal point ``(1.0, 1.0)`` represents top-decile Value and
    top-decile Momentum.
    """
    _require_columns(panel, {"date", "ticker", "close", "pb"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())

    work["_by"] = book_yield(work["pb"])
    work["_mom"] = work.groupby("ticker", sort=False)["close"].transform(
        lambda s: raw_momentum(s, lookback=mom_lookback, skip=mom_skip)
    )
    work["_by_rank"] = cross_sectional_pct_rank(work, "_by")
    work["_mom_rank"] = cross_sectional_pct_rank(work, "_mom")
    work[col] = value_momentum_distance(work["_by_rank"], work["_mom_rank"])

    work = work.drop(columns=["_by", "_mom", "_by_rank", "_mom_rank"])
    return _restore_order(work, original_index)


def add_value_momentum_residual(
    panel: pd.DataFrame,
    *,
    regression_window: int = 252,
    mom_lookback: int = 252,
    mom_skip: int = 21,
    col: str = "val_mom_resid",
) -> pd.DataFrame:
    """
    Return a copy with standardised residual from rolling OLS of
    ``cs_rank(book_yield)`` on ``cs_rank(raw_momentum)`` in ``col``.

    Output is ``residual / rolling_std(residual, regression_window)``;
    NaN when ``std == 0`` or insufficient data.
    """
    if regression_window < 1:
        raise ValueError("regression_window must be >= 1")
    _require_columns(panel, {"date", "ticker", "close", "pb"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())

    work["_by"] = book_yield(work["pb"])
    work["_mom"] = work.groupby("ticker", sort=False)["close"].transform(
        lambda s: raw_momentum(s, lookback=mom_lookback, skip=mom_skip)
    )
    work["_by_rank"] = cross_sectional_pct_rank(work, "_by")
    work["_mom_rank"] = cross_sectional_pct_rank(work, "_mom")

    work[col] = np.nan
    for _, grp in work.groupby("ticker", sort=False):
        y = grp["_by_rank"]
        x = grp["_mom_rank"]
        stats = rolling_ols_stats(y, x, regression_window)
        resid = y.astype(float) - stats["alpha"] - stats["beta"] * x.astype(float)
        rolling_std = resid.rolling(
            window=regression_window, min_periods=regression_window
        ).std(ddof=1)
        standardised = resid / rolling_std.where(rolling_std != 0)
        work.loc[grp.index, col] = standardised.to_numpy()

    work = work.drop(columns=["_by", "_mom", "_by_rank", "_mom_rank"])
    return _restore_order(work, original_index)
