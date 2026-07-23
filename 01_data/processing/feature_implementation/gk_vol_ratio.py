"""H-002 GK vol ratio: series primitives and panel helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
)

VALID_MODES = frozenset({"ratio", "log_ratio", "reversal"})
DEFAULT_GK_WINDOW = 5
DEFAULT_REALISED_WINDOW = 20

_REQUIRED_OHLC = frozenset({"date", "ticker", "open", "high", "low", "close"})
_GK_COEF = 2.0 * np.log(2.0) - 1.0


def garman_klass_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Daily Garman-Klass variance:

    ``0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2``

    Invalid bars (non-positive OHLC or ``H < L``) yield NaN.
    """
    o = open_.astype(float)
    h = high.astype(float)
    lo = low.astype(float)
    c = close.astype(float)
    valid = (o > 0) & (h > 0) & (lo > 0) & (c > 0) & (h >= lo)
    hl = np.log(h / lo)
    co = np.log(c / o)
    var = 0.5 * hl * hl - _GK_COEF * co * co
    return pd.Series(np.where(valid, var, np.nan), index=open_.index, dtype=float)


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Square root of GK variance, clipped at zero when variance is negative."""
    var = garman_klass_variance(open_, high, low, close)
    return np.sqrt(var.clip(lower=0.0))


def log_close_return(close: pd.Series) -> pd.Series:
    """``ln(C_t / C_{t-1})``."""
    c = close.astype(float)
    return np.log(c / c.shift(1))


def realised_vol(
    close: pd.Series,
    window: int = DEFAULT_REALISED_WINDOW,
) -> pd.Series:
    """
    Rolling population std of log close-to-close returns ending at each bar.

    Uses the last ``window`` log returns (needs ``window + 1`` closes for the
    first valid value). ``min_periods=window``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    rets = log_close_return(close)
    return rets.rolling(window=window, min_periods=window).std(ddof=0)


def ratio_from_vols(short_gk: pd.Series, realised: pd.Series) -> pd.Series:
    """``short_gk / realised``; non-positive or non-finite realised -> NaN."""
    sg = short_gk.astype(float)
    rv = realised.astype(float)
    ok = sg.notna() & rv.notna() & np.isfinite(rv) & (rv > 0)
    out = sg / rv
    return out.where(ok)


def apply_ratio_mode(raw_ratio: pd.Series, mode: str) -> pd.Series:
    """Map raw GK/realised ratio through ``mode`` (before optional CS rank)."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    r = raw_ratio.astype(float)
    if mode == "ratio":
        return r
    if mode == "log_ratio":
        return np.log(r.where(r > 0))
    # reversal
    return -r


def add_gk_vol(
    panel: pd.DataFrame,
    *,
    col: str = "gk_vol",
) -> pd.DataFrame:
    """Return a copy of ``panel`` with per-ticker daily GK volatility in ``col``."""
    _require_columns(panel, _REQUIRED_OHLC)
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = garman_klass_vol(work["open"], work["high"], work["low"], work["close"])
    return _restore_order(work, original_index)


def add_gk_vol_mean(
    panel: pd.DataFrame,
    *,
    gk_window: int = DEFAULT_GK_WINDOW,
    gk_col: str = "gk_vol",
    col: str = "gk_vol_mean",
) -> pd.DataFrame:
    """
    Return a copy with the short-window mean of daily GK vol in ``col``.

    Computes GK vol into ``gk_col`` first when that column is missing.
    """
    if gk_window < 1:
        raise ValueError("gk_window must be >= 1")
    _require_columns(panel, _REQUIRED_OHLC)
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    work = panel if gk_col in panel.columns else add_gk_vol(panel, col=gk_col)
    original_index = work.index
    work = _sorted_by_ticker_date(work.copy())
    work[col] = work.groupby("ticker", sort=False)[gk_col].transform(
        lambda s: s.rolling(window=gk_window, min_periods=gk_window).mean()
    )
    return _restore_order(work, original_index)


def add_realised_vol(
    panel: pd.DataFrame,
    *,
    realised_window: int = DEFAULT_REALISED_WINDOW,
    col: str = "realised_vol",
) -> pd.DataFrame:
    """Return a copy with per-ticker close-to-close realised vol in ``col``."""
    if realised_window < 1:
        raise ValueError("realised_window must be >= 1")
    _require_columns(panel, {"date", "ticker", "close"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = work.groupby("ticker", sort=False)["close"].transform(
        lambda s: realised_vol(s, window=realised_window)
    )
    return _restore_order(work, original_index)


def add_gk_realised_ratio_raw(
    panel: pd.DataFrame,
    *,
    gk_window: int = DEFAULT_GK_WINDOW,
    realised_window: int = DEFAULT_REALISED_WINDOW,
    col: str = "gk_realised_ratio_raw",
) -> pd.DataFrame:
    """
    Return a copy with raw short-GK-mean / realised-vol ratio in ``col``.

    Value is **before** mode transform and cross-sectional normalization.
    """
    if gk_window < 1:
        raise ValueError("gk_window must be >= 1")
    if realised_window < 1:
        raise ValueError("realised_window must be >= 1")
    _require_columns(panel, _REQUIRED_OHLC)

    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())

    gk = garman_klass_vol(work["open"], work["high"], work["low"], work["close"])
    gk_mean = gk.groupby(work["ticker"], sort=False).transform(
        lambda s: s.rolling(window=gk_window, min_periods=gk_window).mean()
    )
    rv = work.groupby("ticker", sort=False)["close"].transform(
        lambda s: realised_vol(s, window=realised_window)
    )
    work[col] = ratio_from_vols(gk_mean, rv)
    return _restore_order(work, original_index)
