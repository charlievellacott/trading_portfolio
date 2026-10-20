"""S1 star-stack path, I/O re-exports, and the variant ledger."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

from backtest.star_stack_io import (
    load_star_stack,
    require_star,
    save_star_stack,
    update_star_stack_key,
    vt_target_ann_vol_from_stack,
)
from data.repo_paths import repo_root

ARTIFACTS_DIR = os.path.join(repo_root(), "04_backtest", "s1_equities", "artifacts")
DEFAULT_STAR_STACK = os.path.join(ARTIFACTS_DIR, "s1_star_stack.json")

VARIANT_LEDGER_PATH = os.path.join(ARTIFACTS_DIR, "s1_variant_ledger.json")

HYPOTHESIS_ORDER: tuple[str, ...] = tuple(f"H-{i:03d}" for i in range(1, 16))


def _normalize_hyp_id(hyp_id: str) -> str:
    key = str(hyp_id).strip().upper()
    if key not in HYPOTHESIS_ORDER:
        raise KeyError(f"unknown hyp_id {hyp_id!r}; expected one of {HYPOTHESIS_ORDER}")
    return key


def load_variant_ledger(path: str | None = None) -> dict:
    p = path or VARIANT_LEDGER_PATH
    if not os.path.isfile(p):
        return {"entries": [], "cumulative_arms": 0}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_variant_ledger(payload: dict, path: str | None = None) -> None:
    p = path or VARIANT_LEDGER_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    entries = payload.get("entries") or []
    payload["cumulative_arms"] = int(
        sum(int(e.get("n_arms", len(e.get("arms", [])))) for e in entries)
    )
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def cumulative_trials_before(hyp_id: str, *, path: str | None = None) -> int:
    key = _normalize_hyp_id(hyp_id)
    idx = HYPOTHESIS_ORDER.index(key)
    if idx == 0:
        return 0
    prior = set(HYPOTHESIS_ORDER[:idx])
    ledger = load_variant_ledger(path)
    total = 0
    for entry in ledger.get("entries") or []:
        eid = str(entry.get("hyp_id", "")).strip().upper()
        if eid in prior:
            total += int(entry.get("n_arms", len(entry.get("arms", []))))
    return total


def n_trials_local(configs: dict) -> int:
    return len(configs)


def n_trials_ledger_total(*, path: str | None = None) -> int:
    ledger = load_variant_ledger(path)
    total = 0
    for entry in ledger.get("entries") or []:
        total += int(entry.get("n_arms", len(entry.get("arms", []))))
    return int(total)


def n_trials_stack(hyp_id: str, configs: dict, *, path: str | None = None) -> int:
    return cumulative_trials_before(hyp_id, path=path) + n_trials_local(configs)


def register_hypothesis_arms(
    hyp_id: str,
    arm_names: Sequence[str],
    *,
    overwrite: bool = False,
    path: str | None = None,
) -> dict:
    """Append or replace the ledger entry for a hypothesis screen."""
    key = _normalize_hyp_id(hyp_id)
    arms = [str(a) for a in arm_names]
    ledger = load_variant_ledger(path)
    entries: list[dict] = list(ledger.get("entries") or [])
    row = {"hyp_id": key, "arms": arms, "n_arms": len(arms)}
    replaced = False
    for i, entry in enumerate(entries):
        if str(entry.get("hyp_id", "")).strip().upper() == key:
            if not overwrite:
                raise ValueError(
                    f"{key} already in variant ledger; pass overwrite=True to replace"
                )
            entries[i] = row
            replaced = True
            break
    if not replaced:
        entries.append(row)
    entries.sort(key=lambda e: HYPOTHESIS_ORDER.index(_normalize_hyp_id(e["hyp_id"])))
    ledger["entries"] = entries
    save_variant_ledger(ledger, path)
    return ledger


__all__ = [
    "ARTIFACTS_DIR",
    "DEFAULT_STAR_STACK",
    "HYPOTHESIS_ORDER",
    "VARIANT_LEDGER_PATH",
    "cumulative_trials_before",
    "load_star_stack",
    "load_variant_ledger",
    "n_trials_ledger_total",
    "n_trials_local",
    "n_trials_stack",
    "register_hypothesis_arms",
    "require_star",
    "save_star_stack",
    "save_variant_ledger",
    "update_star_stack_key",
    "vt_target_ann_vol_from_stack",
]
