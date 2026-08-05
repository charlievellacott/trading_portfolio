"""Open-to-open trailing realized volatility primitives and panel helpers."""

from __future__ import annotations

import pandas as pd

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    daily_simple_return,
    normalize_windows,
    windowed_column_name,
)

DEFAULT_OPEN_VOL_WINDOW = 21


# ---------------------------------------------------------------------------
# Series primitives
# ---------------------------------------------------------------------------


def open_to_open_return(open_px: pd.Series) -> pd.Series:
    """``O_t / O_{t-1} - 1``; non-positive / non-finite opens → NaN."""
    return daily_simple_return(open_px)


def trailing_realized_vol(
    returns: pd.Series,
    window: int,
    *,
    ddof: int = 1,
) -> pd.Series:
    """
    Rolling sample stdev of ``returns`` with ``min_periods=window``.

    Does **not** apply a PIT lag; use ``pit_shift_vol`` (or pass
    ``pit_shift`` in matrix/panel helpers) so date ``d`` excludes ``d``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    r = returns.astype(float)
    return r.rolling(window, min_periods=window).std(ddof=ddof)


def pit_shift_vol(vol: pd.Series, *, periods: int = 1) -> pd.Series:
    """
    Lag volatility so the value on date ``d`` uses only information before ``d``.

    For S1 pre-open decisions, ``periods=1`` means trade date ``d`` sees vol
    computed through the prior bar only.
    """
    if periods < 0:
        raise ValueError("periods must be >= 0")
    if periods == 0:
        return vol.astype(float)
    return vol.astype(float).shift(periods)


def open_realized_vol(
    open_px: pd.Series,
    window: int = DEFAULT_OPEN_VOL_WINDOW,
    *,
    pit_shift: int = 1,
    ddof: int = 1,
) -> pd.Series:
    """
    Trailing open-to-open realized vol for one series.

    Default ``pit_shift=1`` makes the value on calendar date ``d`` use only
    open-to-open returns strictly before ``d``.
    """
    rets = open_to_open_return(open_px)
    vol = trailing_realized_vol(rets, window, ddof=ddof)
    return pit_shift_vol(vol, periods=pit_shift)


# ---------------------------------------------------------------------------
# Wide matrix (backtest pivots)
# ---------------------------------------------------------------------------


def trailing_open_vol_matrix(
    opens: pd.DataFrame,
    *,
    window: int = DEFAULT_OPEN_VOL_WINDOW,
    pit_shift: int = 1,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Column-wise open-to-open trailing vol for a wide open-price pivot.

    Index = trading dates, columns = tickers. Same PIT convention as
    ``open_realized_vol``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if pit_shift < 0:
        raise ValueError("pit_shift must be >= 0")

    out = pd.DataFrame(index=opens.index, columns=opens.columns, dtype=float)
    for col in opens.columns:
        out[col] = open_realized_vol(
            opens[col],
            window,
            pit_shift=pit_shift,
            ddof=ddof,
        )
    return out


# ---------------------------------------------------------------------------
# Long-panel helpers
# ---------------------------------------------------------------------------


def add_open_realized_vol(
    panel: pd.DataFrame,
    *,
    window: int = DEFAULT_OPEN_VOL_WINDOW,
    open_col: str = "open",
    col: str = "open_realized_vol",
    pit_shift: int = 1,
    ddof: int = 1,
) -> pd.DataFrame:
    """Attach one open realized-vol column to a long OHLCV panel."""
    if window < 1:
        raise ValueError("window must be >= 1")
    required = {"date", "ticker", open_col}
    _require_columns(panel, required)
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    parts = []
    for _, grp in work.groupby("ticker", sort=False):
        parts.append(
            open_realized_vol(
                grp[open_col],
                window,
                pit_shift=pit_shift,
                ddof=ddof,
            )
        )
    work[col] = pd.concat(parts).reindex(work.index)
    return _restore_order(work, original_index)


def add_open_realized_vol_multi(
    panel: pd.DataFrame,
    *,
    windows: int | list[int] | tuple[int, ...] = DEFAULT_OPEN_VOL_WINDOW,
    open_col: str = "open",
    pit_shift: int = 1,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Add ``open_realized_vol`` [``_{w}``] for each window.

    One window → bare ``open_realized_vol``; multiple → ``open_realized_vol_{w}``.
    """
    window_list = normalize_windows(windows)
    multi = len(window_list) > 1
    result = panel
    for w in window_list:
        col = windowed_column_name("open_realized_vol", w, multi=multi)
        result = add_open_realized_vol(
            result,
            window=w,
            open_col=open_col,
            col=col,
            pit_shift=pit_shift,
            ddof=ddof,
        )
    return result
