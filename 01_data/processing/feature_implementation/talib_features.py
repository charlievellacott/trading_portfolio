"""TA-Lib technical indicators: thin per-ticker wrappers (no hand-rolled formulae)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    normalize_windows,
    regression_column_name,
)

_REQUIRED_OHLC = frozenset({"date", "ticker", "open", "high", "low", "close"})
_REQUIRED_OHLCV = _REQUIRED_OHLC | frozenset({"volume"})


# ---------------------------------------------------------------------------
# Series primitives (single-ticker arrays)
# ---------------------------------------------------------------------------


def rsi_series(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """TA-Lib RSI on close."""
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    arr = talib.RSI(close.astype(float).to_numpy(), timeperiod=timeperiod)
    return pd.Series(arr, index=close.index, dtype=float)


def adx_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14,
) -> pd.Series:
    """TA-Lib ADX."""
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    arr = talib.ADX(
        high.astype(float).to_numpy(),
        low.astype(float).to_numpy(),
        close.astype(float).to_numpy(),
        timeperiod=timeperiod,
    )
    return pd.Series(arr, index=close.index, dtype=float)


def mfi_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    timeperiod: int = 14,
) -> pd.Series:
    """TA-Lib Money Flow Index."""
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    arr = talib.MFI(
        high.astype(float).to_numpy(),
        low.astype(float).to_numpy(),
        close.astype(float).to_numpy(),
        volume.astype(float).to_numpy(),
        timeperiod=timeperiod,
    )
    return pd.Series(arr, index=close.index, dtype=float)


def bb_percent_b_series(
    close: pd.Series,
    timeperiod: int = 20,
    nbdev: float = 2.0,
) -> pd.Series:
    """
    Bollinger %B from TA-Lib ``BBANDS``:
    ``(close - lower) / (upper - lower)``; NaN when band width ``<= 0``.
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    if nbdev <= 0:
        raise ValueError("nbdev must be > 0")
    upper, _mid, lower = talib.BBANDS(
        close.astype(float).to_numpy(),
        timeperiod=timeperiod,
        nbdevup=nbdev,
        nbdevdn=nbdev,
        matype=0,
    )
    width = upper - lower
    pct_b = np.where(width > 0, (close.astype(float).to_numpy() - lower) / width, np.nan)
    return pd.Series(pct_b, index=close.index, dtype=float)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _per_ticker_ohlc(panel: pd.DataFrame, fn) -> pd.Series:
    parts = []
    for _, grp in panel.groupby("ticker", sort=False):
        parts.append(fn(grp))
    return pd.concat(parts).reindex(panel.index)


def add_rsi(
    panel: pd.DataFrame,
    *,
    timeperiod: int = 14,
    col: str = "rsi",
) -> pd.DataFrame:
    """Attach RSI column (raw TA-Lib scale; no CS normalize)."""
    _require_columns(panel, {"date", "ticker", "close"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = _per_ticker_ohlc(
        work, lambda g: rsi_series(g["close"], timeperiod=timeperiod)
    )
    return _restore_order(work, original_index)


def add_adx(
    panel: pd.DataFrame,
    *,
    timeperiod: int = 14,
    col: str = "adx",
) -> pd.DataFrame:
    """Attach ADX column (raw TA-Lib scale; no CS normalize)."""
    _require_columns(panel, {"date", "ticker", "high", "low", "close"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = _per_ticker_ohlc(
        work,
        lambda g: adx_series(
            g["high"], g["low"], g["close"], timeperiod=timeperiod
        ),
    )
    return _restore_order(work, original_index)


def add_mfi(
    panel: pd.DataFrame,
    *,
    timeperiod: int = 14,
    col: str = "mfi",
) -> pd.DataFrame:
    """Attach MFI column (raw TA-Lib scale; no CS normalize)."""
    _require_columns(panel, {"date", "ticker", "high", "low", "close", "volume"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = _per_ticker_ohlc(
        work,
        lambda g: mfi_series(
            g["high"], g["low"], g["close"], g["volume"], timeperiod=timeperiod
        ),
    )
    return _restore_order(work, original_index)


def add_bb_percent_b(
    panel: pd.DataFrame,
    *,
    timeperiod: int = 20,
    nbdev: float = 2.0,
    col: str = "bb_percent_b",
) -> pd.DataFrame:
    """Attach Bollinger %B column (no CS normalize)."""
    _require_columns(panel, {"date", "ticker", "close"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = _per_ticker_ohlc(
        work,
        lambda g: bb_percent_b_series(
            g["close"], timeperiod=timeperiod, nbdev=nbdev
        ),
    )
    return _restore_order(work, original_index)


def add_rsi_multi(
    panel: pd.DataFrame,
    *,
    timeperiod: int | list[int] | tuple[int, ...] = 14,
) -> pd.DataFrame:
    """Add ``rsi`` / ``rsi_{P}`` for each timeperiod."""
    periods = normalize_windows(timeperiod)
    multi = len(periods) > 1
    result = panel
    for p in periods:
        col = regression_column_name("rsi", p, multi_window=multi)
        result = add_rsi(result, timeperiod=p, col=col)
    return result


def add_adx_multi(
    panel: pd.DataFrame,
    *,
    timeperiod: int | list[int] | tuple[int, ...] = 14,
) -> pd.DataFrame:
    """Add ``adx`` / ``adx_{P}`` for each timeperiod."""
    periods = normalize_windows(timeperiod)
    multi = len(periods) > 1
    result = panel
    for p in periods:
        col = regression_column_name("adx", p, multi_window=multi)
        result = add_adx(result, timeperiod=p, col=col)
    return result


def add_mfi_multi(
    panel: pd.DataFrame,
    *,
    timeperiod: int | list[int] | tuple[int, ...] = 14,
) -> pd.DataFrame:
    """Add ``mfi`` / ``mfi_{P}`` for each timeperiod."""
    periods = normalize_windows(timeperiod)
    multi = len(periods) > 1
    result = panel
    for p in periods:
        col = regression_column_name("mfi", p, multi_window=multi)
        result = add_mfi(result, timeperiod=p, col=col)
    return result


def add_bb_percent_b_multi(
    panel: pd.DataFrame,
    *,
    timeperiod: int | list[int] | tuple[int, ...] = 20,
    nbdev: float = 2.0,
) -> pd.DataFrame:
    """Add ``bb_percent_b`` / ``bb_percent_b_{P}`` for each timeperiod."""
    periods = normalize_windows(timeperiod)
    multi = len(periods) > 1
    result = panel
    for p in periods:
        col = regression_column_name("bb_percent_b", p, multi_window=multi)
        result = add_bb_percent_b(
            result, timeperiod=p, nbdev=nbdev, col=col
        )
    return result
