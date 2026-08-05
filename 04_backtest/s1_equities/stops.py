"""Per-name ATR / percentage stop helpers for S1 backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STOP_NONE = "none"
STOP_ATR = "atr"
STOP_PCT = "pct"
VALID_STOP_MODES = frozenset({STOP_NONE, STOP_ATR, STOP_PCT})


@dataclass(frozen=True)
class StopConfig:
    """Stop-loss configuration for one backtest run."""

    mode: str = STOP_NONE
    atr_window: int | None = None
    atr_multiple: float | None = None
    pct: float | None = None

    def label(self) -> str:
        if self.mode == STOP_NONE:
            return STOP_NONE
        if self.mode == STOP_PCT:
            return f"pct_{self.pct:g}"
        return f"atr_{self.atr_window}_{self.atr_multiple:g}"


def parse_stop_star(stop_star: str) -> StopConfig:
    """Parse ``none``, ``atr_{window}_{multiple}``, or ``pct_{percent}``."""
    s = str(stop_star).strip().lower()
    if s == STOP_NONE:
        return StopConfig(mode=STOP_NONE)
    if s.startswith("pct_"):
        parts = s.split("_")
        if len(parts) != 2:
            raise ValueError(f"expected pct_<percent>, got {stop_star!r}")
        pct = float(parts[1])
        if not np.isfinite(pct) or pct <= 0:
            raise ValueError(f"pct stop must be positive, got {stop_star!r}")
        return StopConfig(mode=STOP_PCT, pct=pct)
    if not s.startswith("atr_"):
        raise ValueError(f"unrecognized STOP_STAR={stop_star!r}")
    parts = s.split("_")
    if len(parts) != 3:
        raise ValueError(f"expected atr_<window>_<k>, got {stop_star!r}")
    return StopConfig(
        mode=STOP_ATR,
        atr_window=int(parts[1]),
        atr_multiple=float(parts[2]),
    )


def stop_levels_from_entry(
    entry_open: pd.Series,
    weights: pd.Series,
    atr_row: pd.Series,
    *,
    atr_multiple: float,
) -> pd.Series:
    """
    Per-name stop prices at entry.

    Long (w>0): ``O - k*ATR``; short (w<0): ``O + k*ATR``.
    Missing/non-positive ATR → NaN stop (hold to schedule).
    """
    w = weights.replace(0.0, np.nan).dropna()
    o = entry_open.reindex(w.index).astype(float)
    a = atr_row.reindex(w.index).astype(float)
    stops = pd.Series(np.nan, index=w.index, dtype=float)
    ok = (
        o.notna()
        & a.notna()
        & np.isfinite(o)
        & np.isfinite(a)
        & (o > 0)
        & (a > 0)
    )
    long_m = ok & (w > 0)
    short_m = ok & (w < 0)
    stops.loc[long_m] = o.loc[long_m] - float(atr_multiple) * a.loc[long_m]
    stops.loc[short_m] = o.loc[short_m] + float(atr_multiple) * a.loc[short_m]
    return stops


def stop_levels_from_entry_pct(
    entry_open: pd.Series,
    weights: pd.Series,
    *,
    pct: float,
) -> pd.Series:
    """
    Per-name fixed-percentage stop prices at entry.

    Long (w>0): ``O * (1 - pct/100)``; short (w<0): ``O * (1 + pct/100)``.
    Missing/non-positive open → NaN stop (hold to schedule).
    """
    if not np.isfinite(pct) or pct <= 0:
        raise ValueError(f"pct must be positive, got {pct!r}")
    w = weights.replace(0.0, np.nan).dropna()
    o = entry_open.reindex(w.index).astype(float)
    stops = pd.Series(np.nan, index=w.index, dtype=float)
    ok = o.notna() & np.isfinite(o) & (o > 0)
    frac = float(pct) / 100.0
    long_m = ok & (w > 0)
    short_m = ok & (w < 0)
    stops.loc[long_m] = o.loc[long_m] * (1.0 - frac)
    stops.loc[short_m] = o.loc[short_m] * (1.0 + frac)
    return stops


def gap_aware_stop_fill(
    *,
    side: str,
    open_px: float,
    stop: float,
) -> float:
    """Fill at stop unless the open already gapped through."""
    if side == "long":
        return float(min(open_px, stop)) if open_px < stop else float(stop)
    if side == "short":
        return float(max(open_px, stop)) if open_px > stop else float(stop)
    raise ValueError(f"side must be long/short, got {side!r}")


def apply_stops_over_hold(
    weights: pd.Series,
    *,
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
    entry_open: pd.Series,
    stops: pd.Series,
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    schedule_exit_px: pd.Series,
    exit_on_open: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Index]:
    """
    Path-dependent per-name exits over the hold.

    Intermediate sessions use daily high/low. When ``exit_on_open`` is True
    (Mon→Mon), the exit date is checked only for an open gap through the stop
    (no same-day H/L after the open exit). When False (Mon→Fri close), the exit
    date is included in the H/L path.
    """
    w0 = weights.replace(0.0, np.nan).dropna()
    exit_px = pd.Series(dtype=float)
    live = w0.copy()
    stopped: list[str] = []

    if w0.empty:
        return exit_px, live, pd.Index([])

    cal = opens.index.union(highs.index).union(lows.index).sort_values().unique()
    cal = pd.DatetimeIndex(cal)
    if exit_on_open:
        days = cal[(cal > entry) & (cal < exit_)]
    else:
        days = cal[(cal > entry) & (cal <= exit_)]

    remaining = set(map(str, w0.index))
    for dt in days:
        if not remaining:
            break
        if dt not in lows.index or dt not in highs.index or dt not in opens.index:
            continue
        lo = lows.loc[dt]
        hi = highs.loc[dt]
        op = opens.loc[dt]
        hit_today: list[str] = []
        for ticker in list(remaining):
            stop = stops.get(ticker, np.nan)
            if not np.isfinite(stop):
                continue
            wt = float(live.get(ticker, 0.0))
            if wt == 0.0:
                continue
            o_px = op.get(ticker, np.nan)
            if not np.isfinite(o_px) or o_px <= 0:
                continue
            if wt > 0:
                low_t = lo.get(ticker, np.nan)
                if np.isfinite(low_t) and low_t <= stop:
                    exit_px.loc[ticker] = gap_aware_stop_fill(
                        side="long", open_px=float(o_px), stop=float(stop)
                    )
                    hit_today.append(ticker)
            else:
                high_t = hi.get(ticker, np.nan)
                if np.isfinite(high_t) and high_t >= stop:
                    exit_px.loc[ticker] = gap_aware_stop_fill(
                        side="short", open_px=float(o_px), stop=float(stop)
                    )
                    hit_today.append(ticker)
        for ticker in hit_today:
            remaining.discard(ticker)
            live.loc[ticker] = 0.0
            stopped.append(ticker)

    # Open-exit gap check on exit date (weekend / overnight into next Monday open)
    if exit_on_open and remaining and exit_ in opens.index:
        op = opens.loc[exit_]
        hit_gap: list[str] = []
        for ticker in list(remaining):
            stop = stops.get(ticker, np.nan)
            if not np.isfinite(stop):
                continue
            wt = float(live.get(ticker, 0.0))
            if wt == 0.0:
                continue
            o_px = op.get(ticker, np.nan)
            if not np.isfinite(o_px) or o_px <= 0:
                continue
            if wt > 0 and o_px < stop:
                exit_px.loc[ticker] = float(o_px)
                hit_gap.append(ticker)
            elif wt < 0 and o_px > stop:
                exit_px.loc[ticker] = float(o_px)
                hit_gap.append(ticker)
        for ticker in hit_gap:
            remaining.discard(ticker)
            live.loc[ticker] = 0.0
            stopped.append(ticker)

    # Scheduled exit for survivors
    sched = schedule_exit_px.reindex(live.index)
    for ticker, wt in live.items():
        if float(wt) == 0.0:
            continue
        px = sched.get(ticker, np.nan)
        e0 = entry_open.get(ticker, np.nan)
        if np.isfinite(px) and px > 0 and np.isfinite(e0) and e0 > 0:
            exit_px.loc[ticker] = float(px)
        else:
            live.loc[ticker] = 0.0

    return exit_px, live, pd.Index(stopped)
