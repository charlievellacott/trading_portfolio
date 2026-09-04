"""Sleeve-agnostic star-stack JSON I/O. No strategy-specific sim config here."""

from __future__ import annotations

import json
import os

from data.repo_paths import repo_root

VT_TARGET_ANN_VOL_STAR = "VT_TARGET_ANN_VOL_STAR"

_SLEEVE_FOLDERS = {
    "s1": "s1_equities",
    "s1_equities": "s1_equities",
    "s2": "s2_coint",
    "s2_coint": "s2_coint",
}

_SLEEVE_FILES = {
    "s1": "s1_star_stack.json",
    "s1_equities": "s1_star_stack.json",
    "s2": "s2_star_stack.json",
    "s2_coint": "s2_star_stack.json",
}


def default_star_stack_path(sleeve: str, repo_root_dir: str | None = None) -> str:
    """``04_backtest/{sleeve_folder}/artifacts/{sleeve}_star_stack.json``."""
    key = str(sleeve).strip().lower()
    folder = _SLEEVE_FOLDERS.get(key)
    fname = _SLEEVE_FILES.get(key)
    if folder is None or fname is None:
        raise ValueError(
            f"unknown sleeve {sleeve!r}; expected one of {sorted(_SLEEVE_FOLDERS)}"
        )
    root = repo_root_dir if repo_root_dir is not None else repo_root()
    return os.path.join(root, "04_backtest", folder, "artifacts", fname)


def require_star(name: str, value) -> None:
    if value is None:
        raise ValueError(
            f"{name} is None. Type it in the freeze cell after reviewing fold-val metrics."
        )


def load_star_stack(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"star stack not found: {path}.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_star_stack(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def update_star_stack_key(
    path: str,
    key: str,
    value,
    *,
    extra: dict | None = None,
) -> dict:
    """Read-merge-write one STAR key (and optional extra fields)."""
    if os.path.isfile(path):
        payload = load_star_stack(path)
    else:
        payload = {}
    payload[str(key)] = value
    if extra:
        payload.update(extra)
    save_star_stack(path, payload)
    return payload


def vt_target_ann_vol_from_stack(
    stack: dict,
    *,
    default: float = 0.10,
) -> float:
    raw = stack.get(VT_TARGET_ANN_VOL_STAR)
    if raw is None:
        raw = stack.get("vt_target_ann_vol")
    if raw is None:
        return float(default)
    return float(raw)
