"""H-004 pair-book construction: ranked selection, caps, demotion and orientation locks.

Strategy layer, not research: these rules run in production under ``BOOK_STAR = rotate``.

Pre-registered constants (not searched, not STARs): quarterly cadence, ``L = 252`` trailing
bars for the rotating screen, ``alpha = 0.05``, global cap 6, per-pool cap 2, ``1 / 6`` slot
weighting. No half-life filter before H-008. No minimum tenure - demotion never forces a
trade, so there is nothing to damp.

Core semantics, identical for demotion and orientation flips:

- A pair leaving the active set is **blocked from new entries** and runs any open position to
  its normal z-exit. Nothing is force-closed, so rotation adds no trading cost of its own.
- An unordered pair ``{a, b}`` occupies **one** slot regardless of orientation, so caps are
  never double-counted and a flip cannot smuggle in a second slot.
- When Engle-Granger flips the orientation of an active pair, the old orientation is blocked
  and the new one waits until the old is flat.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from data.processing.feature_implementation.cointegration import COINT_PVALUE

CAP_GLOBAL = 6
CAP_PER_POOL = 2
REBALANCE_FREQ = "quarterly"
DISCOVERY_LOOKBACK_BARS = 252
SLOT_WEIGHT = 1.0 / CAP_GLOBAL

__all__ = [
    "CAP_GLOBAL",
    "CAP_PER_POOL",
    "DISCOVERY_LOOKBACK_BARS",
    "REBALANCE_FREQ",
    "SLOT_WEIGHT",
    "BookState",
    "apply_rebalance",
    "select_book",
    "slot_key",
]


def slot_key(pair_id: str) -> str:
    """Orientation-insensitive slot id: ``"b|a"`` and ``"a|b"`` map to the same slot."""
    parts = str(pair_id).split("|")
    if len(parts) != 2:
        raise ValueError(f"pair_id must be ticker_y|ticker_x, got {pair_id!r}")
    return "|".join(sorted(parts))


def select_book(
    screen: pd.DataFrame,
    *,
    pool_of_pair: Mapping[str, str],
    per_pool_cap: int = CAP_PER_POOL,
    global_cap: int = CAP_GLOBAL,
    pvalue_threshold: float = COINT_PVALUE,
) -> pd.DataFrame:
    """Rank Engle-Granger passers by p-value and apply per-pool then global caps.

    Selection is p-value only: no half-life filter until H-008. Ineligible rows (too few
    mutual bars) and rows at or above ``pvalue_threshold`` are dropped first. Returns the
    kept rows with ``pool`` and ``rank`` columns, ordered best p-value first.
    """
    cols = [*(screen.columns if screen is not None else []), "pool", "rank"]
    if screen is None or screen.empty:
        return pd.DataFrame(columns=cols)

    d = screen.copy()
    if "eligible" in d.columns:
        d = d.loc[d["eligible"].astype(bool)]
    d["pvalue"] = pd.to_numeric(d["pvalue"], errors="coerce")
    d = d.loc[d["pvalue"].notna() & (d["pvalue"] < float(pvalue_threshold))]
    if d.empty:
        return pd.DataFrame(columns=cols)

    d["pool"] = [
        _pool_lookup(pool_of_pair, str(pid)) for pid in d["pair_id"].astype(str)
    ]
    # Stable tie-break on pair_id so equal p-values select deterministically.
    d = d.sort_values(["pvalue", "pair_id"], kind="mergesort").reset_index(drop=True)

    per_pool: dict[str, int] = {}
    used_slots: set[str] = set()
    keep_idx: list[int] = []
    for i, row in enumerate(d.itertuples(index=False)):
        pool = str(row.pool)
        slot = slot_key(str(row.pair_id))
        if slot in used_slots:
            continue
        if per_pool.get(pool, 0) >= int(per_pool_cap):
            continue
        if len(keep_idx) >= int(global_cap):
            break
        keep_idx.append(i)
        per_pool[pool] = per_pool.get(pool, 0) + 1
        used_slots.add(slot)

    out = d.iloc[keep_idx].reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def _pool_lookup(pool_of_pair: Mapping[str, str], pair_id: str) -> str:
    if pair_id in pool_of_pair:
        return pool_of_pair[pair_id]
    parts = pair_id.split("|")
    if len(parts) == 2:
        flipped = f"{parts[1]}|{parts[0]}"
        if flipped in pool_of_pair:
            return pool_of_pair[flipped]
    raise KeyError(f"pair {pair_id!r} is not inside any leaf pool")


@dataclass
class BookState:
    """Active orientations, blocked orientations, and slot occupancy.

    ``active`` maps slot -> the tradable ``pair_id`` orientation. ``blocked`` holds
    orientations that may still exit but must not open. ``pending`` holds an orientation
    waiting for its blocked twin to go flat after a flip.
    """

    active: dict[str, str] = field(default_factory=dict)
    blocked: set[str] = field(default_factory=set)
    pending: dict[str, str] = field(default_factory=dict)

    def active_pairs(self) -> list[str]:
        return sorted(self.active.values())

    def is_entry_allowed(self, pair_id: str) -> bool:
        """True only for the orientation currently holding its slot."""
        slot = slot_key(pair_id)
        return self.active.get(slot) == str(pair_id)

    def release_flat(self, pair_id: str) -> None:
        """Call once a blocked orientation is flat; promotes any pending flip."""
        pid = str(pair_id)
        slot = slot_key(pid)
        self.blocked.discard(pid)
        waiting = self.pending.get(slot)
        if waiting is not None and slot not in self.active:
            self.active[slot] = waiting
            del self.pending[slot]


def apply_rebalance(
    state: BookState,
    selection: pd.DataFrame | list[str],
    *,
    open_pairs: set[str] | None = None,
) -> dict[str, list[str]]:
    """Move the book to ``selection``; return promotions / demotions / flips.

    ``open_pairs`` are orientations with a live position: they become blocked rather than
    dropped, so they can still exit on z. Anything flat is dropped outright at no cost.
    """
    if isinstance(selection, pd.DataFrame):
        wanted = [str(p) for p in selection.get("pair_id", pd.Series(dtype=str))]
    else:
        wanted = [str(p) for p in selection]
    live = set(open_pairs or set())

    wanted_by_slot: dict[str, str] = {}
    for pid in wanted:
        wanted_by_slot.setdefault(slot_key(pid), pid)

    promoted: list[str] = []
    demoted: list[str] = []
    flipped: list[str] = []

    for slot, current in list(state.active.items()):
        target = wanted_by_slot.get(slot)
        if target == current:
            continue
        # Slot lost, or kept but with the other orientation: current stops entering.
        del state.active[slot]
        if current in live:
            state.blocked.add(current)
        if target is None:
            demoted.append(current)
            state.pending.pop(slot, None)
        else:
            flipped.append(f"{current}->{target}")
            if current in live:
                # New orientation waits until the old one is flat.
                state.pending[slot] = target
            else:
                state.active[slot] = target

    for slot, target in wanted_by_slot.items():
        if slot in state.active or slot in state.pending:
            continue
        state.active[slot] = target
        promoted.append(target)

    return {
        "promoted": sorted(promoted),
        "demoted": sorted(demoted),
        "flipped": sorted(flipped),
    }
