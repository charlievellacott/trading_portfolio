"""Wilder ATR (average true range) primitives and panel helpers.

Wilder ATR is the standard smoother used for stop/chandelier rules:
``ATR_t = (ATR_{t-1} * (n - 1) + TR_t) / n``, seeded with the SMA of the
first ``n`` true ranges. Compared with a plain rolling mean of true range
(SMA-TR), Wilder ATR is more persistent and less jumpy — better suited to
stop levels that should not whip around on a single wide bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    normalize_windows,
    windowed_column_name,
)

DEFAULT_ATR_WINDOW = 14


# ---------------------------------------------------------------------------
# Series primitives
# ---------------------------------------------------------------------------


def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    True range: ``max(H-L, |H-prev_C|, |L-prev_C|)``.

    Non-finite / invalid OHLC yield NaN for that bar. On the first bar
    (no prior close), TR reduces to ``H - L`` when valid.
    """
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    prev_c = c.shift(1)
    hl = h - l
    hc = (h - prev_c).abs()
    lc = (l - prev_c).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    ok = (
        h.notna()
        & l.notna()
        & np.isfinite(h)
        & np.isfinite(l)
        & (h > 0)
        & (l > 0)
        & (h >= l)
    )
    first = prev_c.isna()
    ok_rest = (
        ok
        & c.notna()
        & np.isfinite(c)
        & (c > 0)
        & prev_c.notna()
        & np.isfinite(prev_c)
    )
    out = pd.Series(np.nan, index=high.index, dtype=float)
    out.loc[first & ok] = hl.loc[first & ok]
    out.loc[~first & ok_rest] = tr.loc[~first & ok_rest]
    return out


def wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = DEFAULT_ATR_WINDOW,
    *,
    pit_shift: int = 1,
) -> pd.Series:
    """
    Wilder ATR for one series.

    Default ``pit_shift=1`` so the value on date ``d`` uses only bars before
    ``d`` (S1 pre-open).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if pit_shift < 0:
        raise ValueError("pit_shift must be >= 0")

    tr = true_range(high, low, close).to_numpy(dtype=float)
    n = len(tr)
    out = np.full(n, np.nan, dtype=float)

    i = 0
    while i < n:
        if not np.isfinite(tr[i]):
            i += 1
            continue
        if i + window > n:
            break
        chunk = tr[i : i + window]
        if not np.isfinite(chunk).all():
            i += 1
            continue
        prev = float(np.mean(chunk))
        out[i + window - 1] = prev
        j = i + window
        while j < n and np.isfinite(tr[j]):
            prev = (prev * (window - 1) + tr[j]) / window
            out[j] = prev
            j += 1
        i = j if j > i else i + 1

    atr = pd.Series(out, index=high.index, dtype=float)
    if pit_shift:
        atr = atr.shift(pit_shift)
    return atr


# ---------------------------------------------------------------------------
# Wide matrix (backtest pivots)
# ---------------------------------------------------------------------------


def wilder_atr_matrix(
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    window: int = DEFAULT_ATR_WINDOW,
    pit_shift: int = 1,
) -> pd.DataFrame:
    """Column-wise Wilder ATR for wide H/L/C pivots (index=date, columns=ticker)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    cols = highs.columns.intersection(lows.columns).intersection(closes.columns)
    idx = highs.index.intersection(lows.index).intersection(closes.index)
    out = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for col in cols:
        out[col] = wilder_atr(
            highs.loc[idx, col],
            lows.loc[idx, col],
            closes.loc[idx, col],
            window,
            pit_shift=pit_shift,
        )
    return out


# ---------------------------------------------------------------------------
# Long-panel helpers
# ---------------------------------------------------------------------------


def add_wilder_atr(
    panel: pd.DataFrame,
    *,
    window: int = DEFAULT_ATR_WINDOW,
    col: str = "atr",
    pit_shift: int = 1,
) -> pd.DataFrame:
    """Attach one Wilder ATR column to a long OHLCV panel."""
    if window < 1:
        raise ValueError("window must be >= 1")
    _require_columns(panel, {"date", "ticker", "high", "low", "close"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    parts = []
    for _, grp in work.groupby("ticker", sort=False):
        parts.append(
            wilder_atr(
                grp["high"],
                grp["low"],
                grp["close"],
                window,
                pit_shift=pit_shift,
            )
        )
    work[col] = pd.concat(parts).reindex(work.index)
    return _restore_order(work, original_index)


def add_wilder_atr_multi(
    panel: pd.DataFrame,
    *,
    windows: int | list[int] | tuple[int, ...] = DEFAULT_ATR_WINDOW,
    pit_shift: int = 1,
) -> pd.DataFrame:
    """Add ``atr`` [``_{w}``] for each window."""
    window_list = normalize_windows(windows)
    multi = len(window_list) > 1
    result = panel
    for w in window_list:
        col = windowed_column_name("atr", w, multi=multi)
        result = add_wilder_atr(
            result, window=w, col=col, pit_shift=pit_shift
        )
    return result
