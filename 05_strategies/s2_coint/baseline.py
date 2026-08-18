"""H-001 traditional z-score pair simulation (signal close t, fill open t+1)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.s2_coint.costs import leg_cost_bps, market_profile_for_pair

ENTRY_Z = 2.0
EXIT_Z = 0.0
USE_HEDGE_RATIO_SIZING = True
BETA_COLUMN = "beta"
PERIODS_PER_YEAR = 252.0

_TRADE_COLS: tuple[str, ...] = (
    "pair_id",
    "side",
    "entry_date",
    "exit_date",
    "hold_bars",
    "entry_cost_bps",
    "exit_cost_bps",
)


@dataclass(frozen=True)
class PairSimResult:
    """Net daily returns plus completed round-trips for one pair."""

    pair_id: str
    returns: pd.Series
    trades: pd.DataFrame
    n_entries: int
    n_open_at_end: int
    open_entry_cost_bps: float


def clip_ohlc_to_is(
    ohlc_by_ticker: Mapping[str, pd.DataFrame],
    is_end: pd.Timestamp | str,
) -> dict[str, pd.DataFrame]:
    """Keep bars with index date <= research IS end (no OOS rows in the panel)."""
    end = pd.Timestamp(is_end)
    out: dict[str, pd.DataFrame] = {}
    for ticker, frame in ohlc_by_ticker.items():
        idx = pd.to_datetime(frame.index)
        out[ticker] = frame.loc[idx <= end].copy()
    return out


def simulate_pair_baseline(
    df: pd.DataFrame,
    *,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
    beta_column: str = BETA_COLUMN,
    use_hedge_ratio_sizing: bool = USE_HEDGE_RATIO_SIZING,
) -> PairSimResult:
    """Trad-z baseline: decide at close t, fill both legs at open t+1.

    Long the spread when ``z <= -entry_z``; short when ``z >= entry_z``. Exit is
    a signed recross of ``exit_z`` (default 0 = mean): flatten a long when
    ``z >= exit_z``, a short when ``z <= -exit_z``.

    Costs hit on the fill date. Holding PnL is open t+1 → open t+2, attributed
    to the fill date. Completed round-trips populate ``trades``; an open
    position at the last evaluable bar is counted in ``n_open_at_end`` only.
    """
    empty_trades = pd.DataFrame(columns=list(_TRADE_COLS))
    if df.empty:
        return PairSimResult(
            pair_id="",
            returns=pd.Series(dtype=float),
            trades=empty_trades,
            n_entries=0,
            n_open_at_end=0,
            open_entry_cost_bps=0.0,
        )

    d = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(d["date"]).to_list()
    z = d["z"].to_numpy(dtype=float)
    beta = d[beta_column].to_numpy(dtype=float)
    oy = d["open_y"].to_numpy(dtype=float)
    ox = d["open_x"].to_numpy(dtype=float)
    ty = str(d["ticker_y"].iloc[0])
    tx = str(d["ticker_x"].iloc[0])
    pair_id = str(d["pair_id"].iloc[0])

    profile = market_profile_for_pair(pair_id, ty, tx)
    pos = 0  # +1 long spread, -1 short spread
    pnl_by_date: dict[pd.Timestamp, float] = defaultdict(float)
    trades: list[dict] = []
    n_entries = 0
    open_entry: dict | None = None

    for i in range(len(d) - 2):
        z_t = z[i]
        beta_fill = beta[i + 1]
        fill_date = pd.Timestamp(dates[i + 1])

        do_entry = pos == 0 and np.isfinite(z_t) and np.isfinite(beta_fill)
        do_exit = (
            pos != 0
            and np.isfinite(z_t)
            and (float(pos) * z_t >= float(exit_z))
        )

        event_cost = 0.0
        if do_exit:
            by = leg_cost_bps(profile, ty, oy[i + 1])
            bx = leg_cost_bps(profile, tx, ox[i + 1])
            exit_cost_bps = float(by + bx)
            event_cost += exit_cost_bps / 10_000.0
            if open_entry is not None:
                entry_idx = int(open_entry["entry_idx"])
                trades.append(
                    {
                        "pair_id": pair_id,
                        "side": int(open_entry["side"]),
                        "entry_date": open_entry["entry_date"],
                        "exit_date": fill_date,
                        "hold_bars": int((i + 1) - entry_idx),
                        "entry_cost_bps": float(open_entry["entry_cost_bps"]),
                        "exit_cost_bps": exit_cost_bps,
                    }
                )
            open_entry = None
            pos = 0

        if do_entry:
            if z_t >= entry_z:
                pos = -1
            elif z_t <= -entry_z:
                pos = 1
            if pos != 0:
                by = leg_cost_bps(profile, ty, oy[i + 1])
                bx = leg_cost_bps(profile, tx, ox[i + 1])
                entry_cost_bps = float(by + bx)
                event_cost += entry_cost_bps / 10_000.0
                n_entries += 1
                open_entry = {
                    "side": pos,
                    "entry_idx": i + 1,
                    "entry_date": fill_date,
                    "entry_cost_bps": entry_cost_bps,
                }

        bar_ret = 0.0
        if pos != 0 and np.isfinite(beta_fill):
            ry = oy[i + 2] / oy[i + 1] - 1.0
            rx = ox[i + 2] / ox[i + 1] - 1.0
            y_w = float(pos)
            x_w = (
                -float(pos) * float(beta_fill) if use_hedge_ratio_sizing else -float(pos)
            )
            gross = abs(y_w) + abs(x_w)
            if gross > 0:
                y_w /= gross
                x_w /= gross
            bar_ret = y_w * ry + x_w * rx

        pnl_by_date[fill_date] += bar_ret - event_cost

    out = pd.Series(pnl_by_date, dtype=float).sort_index()
    out.name = pair_id
    trade_df = pd.DataFrame(trades, columns=list(_TRADE_COLS))
    open_cost = float(open_entry["entry_cost_bps"]) if open_entry is not None else 0.0
    return PairSimResult(
        pair_id=pair_id,
        returns=out,
        trades=trade_df,
        n_entries=n_entries,
        n_open_at_end=int(open_entry is not None),
        open_entry_cost_bps=open_cost,
    )


def combine_universe_returns(panel: pd.DataFrame, **sim_kwargs) -> pd.Series:
    """Equal-weight net returns across pairs on the supplied panel (caller clips IS)."""
    if panel.empty:
        return pd.Series(dtype=float, name="ret")
    parts: list[pd.Series] = []
    for _, g in panel.groupby("pair_id", sort=False):
        result = simulate_pair_baseline(g, **sim_kwargs)
        if not result.returns.empty:
            parts.append(result.returns.rename(result.pair_id))
    if not parts:
        return pd.Series(dtype=float, name="ret")
    wide = pd.concat(parts, axis=1).fillna(0.0)
    return wide.mean(axis=1).rename("ret")
