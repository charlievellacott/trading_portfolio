"""H-004 research driver: quarterly rebalance schedule and the two book-construction arms.

Backtest layer. The live rules (caps, demotion, orientation locks) live in
``strategies.s2_coint.book``; this module schedules them across research dates and runs the
simulations.

Both arms read the **same** union panel and share the health gate, soft defaults, caps, beta
hedging and slot weighting, so the only difference is how the book is chosen:

- ``freeze``: one Engle-Granger screen on the **full IS** at ``RESEARCH_IS_END``, book fixed
  for the whole evaluation. Full IS because that is what freezing means here - "cointegrated
  in general".
- ``rotate``: Engle-Granger at every quarter-end on the trailing ``L = 252`` bars, active set
  effective from the **next session's open**.

Weighting is a fixed ``1 / CAP_GLOBAL`` per slot in both arms, cash otherwise, so a one-pair
quarter cannot run six times the per-pair risk of a six-pair quarter and confound the
comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from data.processing.s2_coint_store import screen_pair_cointegration
from data.processing.s2_universe import iter_pool_pairs, pool_of_pair
from strategies.s2_coint.book import (
    CAP_GLOBAL,
    CAP_PER_POOL,
    DISCOVERY_LOOKBACK_BARS,
    SLOT_WEIGHT,
    BookState,
    apply_rebalance,
    select_book,
    slot_key,
)
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.engine import simulate_pair
from strategies.s2_coint.metrics import metrics_from_returns
from strategies.s2_coint.short_bans import pair_entry_masks

_SCHEDULE_COLS: tuple[str, ...] = (
    "rebalance_date",
    "effective_date",
    "pair_id",
    "pool",
    "pvalue",
    "rank",
)

__all__ = [
    "SLOT_WEIGHT",
    "build_active_schedule",
    "freeze_book",
    "quarter_end_rebalance_dates",
    "schedule_to_masks",
    "simulate_frozen_book",
    "simulate_rotating_book",
]


def quarter_end_rebalance_dates(
    dates: Sequence,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> list[pd.Timestamp]:
    """Last available session of each Mar / Jun / Sep / Dec quarter.

    Uses actual panel sessions, so a rebalance never lands on a non-trading calendar date.
    The final quarter is included only when it has at least one session.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)))).sort_values().unique()
    idx = pd.DatetimeIndex(idx)
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    if len(idx) == 0:
        return []
    frame = pd.DataFrame({"date": idx})
    frame["quarter"] = frame["date"].dt.to_period("Q")
    return [pd.Timestamp(d) for d in frame.groupby("quarter")["date"].max().tolist()]


def _next_session(dates: pd.DatetimeIndex, after: pd.Timestamp) -> pd.Timestamp | None:
    later = dates[dates > pd.Timestamp(after)]
    return pd.Timestamp(later[0]) if len(later) else None


def freeze_book(
    closes: Mapping[str, pd.Series],
    pools,
    *,
    is_end: pd.Timestamp | str,
    ols_window: int = 252,
    per_pool_cap: int = CAP_PER_POOL,
    global_cap: int = CAP_GLOBAL,
    pvalue_threshold: float = 0.05,
) -> pd.DataFrame:
    """One-shot ranked book at ``is_end`` screened on the **full IS** (no trailing window)."""
    candidates = iter_pool_pairs(pools)
    screen = screen_pair_cointegration(
        closes,
        candidates,
        is_end=is_end,
        ols_window=ols_window,
        pvalue_threshold=pvalue_threshold,
    )
    return select_book(
        screen,
        pool_of_pair=pool_of_pair(pools),
        per_pool_cap=per_pool_cap,
        global_cap=global_cap,
        pvalue_threshold=pvalue_threshold,
    )


def build_active_schedule(
    closes: Mapping[str, pd.Series],
    pools,
    dates: Sequence,
    *,
    lookback_bars: int = DISCOVERY_LOOKBACK_BARS,
    ols_window: int = 252,
    per_pool_cap: int = CAP_PER_POOL,
    global_cap: int = CAP_GLOBAL,
    pvalue_threshold: float = 0.05,
    rebalance_dates: Sequence | None = None,
) -> pd.DataFrame:
    """Quarterly selections for the rotating arm.

    At each rebalance ``T`` the screen sees only ``date <= T`` (trailing ``lookback_bars``),
    and the chosen set becomes effective at the **next** session's open.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)))).sort_values().unique()
    idx = pd.DatetimeIndex(idx)
    rebals = (
        [pd.Timestamp(d) for d in rebalance_dates]
        if rebalance_dates is not None
        else quarter_end_rebalance_dates(idx)
    )
    candidates = iter_pool_pairs(pools)
    pools_by_pair = pool_of_pair(pools)

    rows: list[dict] = []
    for t in rebals:
        effective = _next_session(idx, t)
        if effective is None:
            continue  # No session after T: nothing could be traded on this selection.
        screen = screen_pair_cointegration(
            closes,
            candidates,
            is_end=t,
            ols_window=ols_window,
            pvalue_threshold=pvalue_threshold,
            lookback_bars=lookback_bars,
        )
        chosen = select_book(
            screen,
            pool_of_pair=pools_by_pair,
            per_pool_cap=per_pool_cap,
            global_cap=global_cap,
            pvalue_threshold=pvalue_threshold,
        )
        for row in chosen.itertuples(index=False):
            rows.append(
                {
                    "rebalance_date": t,
                    "effective_date": effective,
                    "pair_id": str(row.pair_id),
                    "pool": str(row.pool),
                    "pvalue": float(row.pvalue),
                    "rank": int(row.rank),
                }
            )
    return pd.DataFrame(rows, columns=list(_SCHEDULE_COLS))


def schedule_to_masks(
    schedule: pd.DataFrame,
    panel_dates: Sequence,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Per-orientation entry masks plus a per-rebalance promotion/demotion log.

    Walks rebalances in order through ``BookState``, so demotions and orientation flips only
    ever block **new** entries; open positions are left to exit on z. Because a demoted pair
    is never force-closed, rotation adds no trading cost of its own, and the mask is all we
    need to express it.

    ``open_pairs`` is unknown at schedule time, so every displaced orientation is treated as
    potentially open (conservative): it is blocked, and a flipped orientation waits one
    rebalance before becoming tradable.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(panel_dates)))).sort_values().unique()
    idx = pd.DatetimeIndex(idx)
    n = len(idx)

    all_pairs = sorted({str(p) for p in schedule.get("pair_id", pd.Series(dtype=str))})
    masks: dict[str, np.ndarray] = {p: np.zeros(n, dtype=bool) for p in all_pairs}
    if schedule.empty or n == 0:
        return masks, pd.DataFrame(
            columns=["rebalance_date", "effective_date", "n_active", "promoted", "demoted", "flipped"]
        )

    state = BookState()
    log: list[dict] = []
    groups = list(schedule.groupby("effective_date", sort=True))

    for gi, (effective, chunk) in enumerate(groups):
        selection = [str(p) for p in chunk["pair_id"]]
        # Conservative: assume anything losing its slot may still hold a position.
        open_pairs = set(state.active.values())
        moves = apply_rebalance(state, selection, open_pairs=open_pairs)

        start = int(idx.searchsorted(pd.Timestamp(effective), side="left"))
        stop = (
            int(idx.searchsorted(pd.Timestamp(groups[gi + 1][0]), side="left"))
            if gi + 1 < len(groups)
            else n
        )
        for pid in state.active.values():
            if pid not in masks:
                masks[pid] = np.zeros(n, dtype=bool)
            masks[pid][start:stop] = True

        log.append(
            {
                "rebalance_date": pd.Timestamp(chunk["rebalance_date"].iloc[0]),
                "effective_date": pd.Timestamp(effective),
                "n_active": len(state.active),
                "promoted": ",".join(moves["promoted"]),
                "demoted": ",".join(moves["demoted"]),
                "flipped": ",".join(moves["flipped"]),
            }
        )

    return masks, pd.DataFrame(log)


def _pair_slice(panel: pd.DataFrame, pair_id: str) -> pd.DataFrame:
    g = panel.loc[panel["pair_id"].astype(str) == str(pair_id)].copy()
    return g.sort_values("date").reset_index(drop=True)


def _combined_masks(
    g: pd.DataFrame,
    *,
    rotation_mask: np.ndarray | None,
    apply_short_bans: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """AND the rotation mask with direction-specific short-ban masks."""
    n = len(g)
    base = np.ones(n, dtype=bool) if rotation_mask is None else rotation_mask.astype(bool)
    if not apply_short_bans:
        return base, base
    ty = str(g["ticker_y"].iloc[0])
    tx = str(g["ticker_x"].iloc[0])
    allow_long, allow_short = pair_entry_masks(ty, tx, g["date"].tolist())
    return base & allow_long, base & allow_short


def _weighted_book(
    per_pair: dict[str, pd.Series],
    *,
    slot_weight: float,
) -> pd.Series:
    """Sum per-pair returns at a fixed slot weight (cash for unused slots)."""
    if not per_pair:
        return pd.Series(dtype=float, name="ret")
    wide = pd.concat(
        [s.rename(pid) for pid, s in per_pair.items() if not s.empty], axis=1
    )
    if wide.empty:
        return pd.Series(dtype=float, name="ret")
    return (wide.fillna(0.0).sum(axis=1) * float(slot_weight)).rename("ret")


def simulate_frozen_book(
    panel: pd.DataFrame,
    locked_pairs: Sequence[str],
    cfg: S2SimConfig | None = None,
    *,
    slot_weight: float = SLOT_WEIGHT,
    apply_short_bans: bool = True,
) -> dict:
    """``freeze`` arm: fixed book, fixed slot weight, short bans still enforced."""
    cfg = cfg or S2SimConfig()
    per_pair: dict[str, pd.Series] = {}
    trades: list[pd.DataFrame] = []
    for pid in locked_pairs:
        g = _pair_slice(panel, pid)
        if g.empty:
            continue
        allow_long, allow_short = _combined_masks(
            g, rotation_mask=None, apply_short_bans=apply_short_bans
        )
        res = simulate_pair(
            g,
            cfg,
            long_entry_allowed=allow_long,
            short_entry_allowed=allow_short,
        )
        per_pair[str(pid)] = res.returns
        if not res.trades.empty:
            trades.append(res.trades)
    ret = _weighted_book(per_pair, slot_weight=slot_weight)
    return {
        "arm": "freeze",
        "returns": ret,
        "pair_returns": per_pair,
        "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        "metrics": metrics_from_returns(ret),
    }


def simulate_rotating_book(
    panel: pd.DataFrame,
    schedule: pd.DataFrame,
    cfg: S2SimConfig | None = None,
    *,
    slot_weight: float = SLOT_WEIGHT,
    apply_short_bans: bool = True,
) -> dict:
    """``rotate`` arm: quarterly active set via entry masks, fixed slot weight."""
    cfg = cfg or S2SimConfig()
    if panel.empty or schedule.empty:
        return {
            "arm": "rotate",
            "returns": pd.Series(dtype=float, name="ret"),
            "pair_returns": {},
            "trades": pd.DataFrame(),
            "metrics": metrics_from_returns(pd.Series(dtype=float)),
            "rebalance_log": pd.DataFrame(),
        }

    dates = pd.DatetimeIndex(
        pd.to_datetime(panel["date"]).sort_values().unique()
    )
    masks, log = schedule_to_masks(schedule, dates)

    per_pair: dict[str, pd.Series] = {}
    trades: list[pd.DataFrame] = []
    for pid, mask in masks.items():
        g = _pair_slice(panel, pid)
        if g.empty:
            continue
        # Panels are unbalanced (each pair starts when both legs have a close), so map the
        # calendar-wide mask onto this pair's own rows.
        pos = dates.searchsorted(pd.DatetimeIndex(pd.to_datetime(g["date"])))
        pair_mask = mask[np.clip(pos, 0, len(mask) - 1)]
        if not pair_mask.any():
            continue
        allow_long, allow_short = _combined_masks(
            g, rotation_mask=pair_mask, apply_short_bans=apply_short_bans
        )
        res = simulate_pair(
            g,
            cfg,
            long_entry_allowed=allow_long,
            short_entry_allowed=allow_short,
        )
        per_pair[str(pid)] = res.returns
        if not res.trades.empty:
            trades.append(res.trades)

    ret = _weighted_book(per_pair, slot_weight=slot_weight)
    return {
        "arm": "rotate",
        "returns": ret,
        "pair_returns": per_pair,
        "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        "metrics": metrics_from_returns(ret),
        "rebalance_log": log,
    }


def book_composition(schedule: pd.DataFrame) -> pd.DataFrame:
    """Active pairs per pool per rebalance - surfaces single-factor concentration."""
    if schedule.empty:
        return pd.DataFrame(columns=["effective_date", "pool", "n_pairs"])
    grouped = (
        schedule.groupby(["effective_date", "pool"], sort=True)["pair_id"]
        .count()
        .reset_index()
        .rename(columns={"pair_id": "n_pairs"})
    )
    return grouped


def slot_keys(pair_ids: Sequence[str]) -> list[str]:
    """Orientation-insensitive slot ids, for turnover accounting."""
    return [slot_key(p) for p in pair_ids]
