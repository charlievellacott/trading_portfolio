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
    "annotate_screen",
    "apply_rebalance",
    "select_book",
    "slot_key",
    "validate_manual_book",
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


def annotate_screen(
    screen: pd.DataFrame,
    *,
    pool_of_pair: Mapping[str, str],
    per_pool_cap: int = CAP_PER_POOL,
    global_cap: int = CAP_GLOBAL,
    pvalue_threshold: float = COINT_PVALUE,
) -> pd.DataFrame:
    """Tag every screen row with auto-selection outcome (same walk as ``select_book``).

    Adds ``pool``, ``auto_selected``, and ``exclude_reason``:
    ``ineligible`` / ``not_passer`` / ``pool_cap`` / ``global_cap`` / ``""`` (kept).
    Does not drop rows — inspection helper for H-001.
    """
    if screen is None or screen.empty:
        out = pd.DataFrame(
            columns=[
                *(screen.columns if screen is not None else []),
                "pool",
                "auto_selected",
                "exclude_reason",
            ]
        )
        return out

    d = screen.copy()
    d["pvalue"] = pd.to_numeric(d["pvalue"], errors="coerce")
    d["pool"] = [
        _pool_lookup(pool_of_pair, str(pid)) for pid in d["pair_id"].astype(str)
    ]
    d["auto_selected"] = False
    d["exclude_reason"] = ""

    eligible = (
        d["eligible"].astype(bool)
        if "eligible" in d.columns
        else pd.Series(True, index=d.index)
    )
    d.loc[~eligible, "exclude_reason"] = "ineligible"
    not_passer = eligible & (d["pvalue"].isna() | (d["pvalue"] >= float(pvalue_threshold)))
    d.loc[not_passer, "exclude_reason"] = "not_passer"

    # Same order as select_book among eligible passers.
    passer_mask = eligible & d["pvalue"].notna() & (d["pvalue"] < float(pvalue_threshold))
    order = (
        d.loc[passer_mask]
        .sort_values(["pvalue", "pair_id"], kind="mergesort")
        .index.tolist()
    )

    per_pool: dict[str, int] = {}
    used_slots: set[str] = set()
    n_kept = 0
    for idx in order:
        pool = str(d.at[idx, "pool"])
        slot = slot_key(str(d.at[idx, "pair_id"]))
        if slot in used_slots:
            # Same unordered pair already kept under the other orientation.
            d.at[idx, "exclude_reason"] = "pool_cap"
            continue
        if per_pool.get(pool, 0) >= int(per_pool_cap):
            d.at[idx, "exclude_reason"] = "pool_cap"
            continue
        if n_kept >= int(global_cap):
            d.at[idx, "exclude_reason"] = "global_cap"
            continue
        d.at[idx, "auto_selected"] = True
        d.at[idx, "exclude_reason"] = ""
        per_pool[pool] = per_pool.get(pool, 0) + 1
        used_slots.add(slot)
        n_kept += 1

    return d.reset_index(drop=True)


def validate_manual_book(
    pair_ids: list[str] | tuple[str, ...],
    screen: pd.DataFrame,
    *,
    pool_of_pair: Mapping[str, str],
    per_pool_cap: int = CAP_PER_POOL,
    global_cap: int = CAP_GLOBAL,
    pvalue_threshold: float = COINT_PVALUE,
) -> pd.DataFrame:
    """Validate a documented manual freeze book; return ranked rows from ``screen``.

    Raises if any id is missing / not an EG passer, if an unordered slot is duplicated,
    or if per-pool / global caps are breached. Fewer than ``global_cap`` pairs is allowed.
    Empty ``pair_ids`` raises (use ``None`` for shelved universes, not an empty list).
    """
    ids = [str(p) for p in pair_ids]
    if not ids:
        raise ValueError(
            "manual book is empty — fill MANUAL_BOOK with EG passers, "
            "or use None for shelved universes"
        )
    if len(ids) > int(global_cap):
        raise ValueError(
            f"manual book has {len(ids)} pairs; global cap is {global_cap}"
        )

    if screen is None or screen.empty:
        raise ValueError("screen is empty; cannot validate manual book")

    d = screen.copy()
    d["pvalue"] = pd.to_numeric(d["pvalue"], errors="coerce")
    by_id = {str(r.pair_id): r for r in d.itertuples(index=False)}
    # Also index flipped orientations so a manual id can match either EG direction.
    by_slot: dict[str, object] = {}
    for r in d.itertuples(index=False):
        by_slot.setdefault(slot_key(str(r.pair_id)), r)

    used_slots: set[str] = set()
    per_pool: dict[str, int] = {}
    rows: list[dict] = []

    for pid in ids:
        slot = slot_key(pid)
        if slot in used_slots:
            raise ValueError(f"duplicate slot in manual book: {pid!r} ({slot})")
        row = by_id.get(pid) or by_slot.get(slot)
        if row is None:
            raise ValueError(f"pair {pid!r} is not in the screen")
        eligible = True
        if hasattr(row, "eligible"):
            eligible = bool(row.eligible)
        pval = float(row.pvalue) if pd.notna(row.pvalue) else float("nan")
        if (not eligible) or (not (pval < float(pvalue_threshold))):
            raise ValueError(
                f"pair {pid!r} is not an EG passer "
                f"(eligible={eligible}, pvalue={pval})"
            )
        pool = _pool_lookup(pool_of_pair, str(row.pair_id))
        if per_pool.get(pool, 0) >= int(per_pool_cap):
            raise ValueError(
                f"per-pool cap {per_pool_cap} breached for pool {pool!r} "
                f"(adding {pid!r})"
            )
        per_pool[pool] = per_pool.get(pool, 0) + 1
        used_slots.add(slot)
        rows.append(
            {
                "pair_id": str(row.pair_id),
                "ticker_y": getattr(row, "ticker_y", str(row.pair_id).split("|")[0]),
                "ticker_x": getattr(row, "ticker_x", str(row.pair_id).split("|")[1]),
                "pvalue": pval,
                "discovery_half_life": float(
                    getattr(row, "discovery_half_life", float("nan"))
                ),
                "n_is_bars": int(getattr(row, "n_is_bars", 0) or 0),
                "eligible": True,
                "pool": pool,
            }
        )

    out = pd.DataFrame(rows)
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
