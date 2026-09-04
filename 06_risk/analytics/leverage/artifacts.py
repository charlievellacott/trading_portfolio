"""Load desk vol-target pick from the sleeve star stack (legacy JSON shape)."""

from __future__ import annotations

import json
import os

from backtest.star_stack_io import (
    default_star_stack_path,
    vt_target_ann_vol_from_stack,
)
from risk.analytics.monte_carlo.loaders import find_repo_root


def artifact_path(repo_root: str, sleeve: str) -> str:
    """Star-stack path for ``sleeve`` (kept name for prop-firm notebooks)."""
    return default_star_stack_path(sleeve, repo_root_dir=repo_root)


def load_leverage_artifact(
    repo_root: str | None = None,
    sleeve: str = "s1",
    *,
    path: str | None = None,
) -> dict | None:
    """Return a dict with ``target_ann_vol`` / ``vt_star``, or ``None`` if missing."""
    root = find_repo_root(repo_root) if path is None else repo_root
    loc = path if path is not None else default_star_stack_path(sleeve, repo_root_dir=str(root))
    if not os.path.isfile(loc):
        return None
    with open(loc, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return None
    if "VT_TARGET_ANN_VOL_STAR" in raw or "N_STAR" in raw or "PAIRS_STAR" in raw:
        return {
            "sleeve": sleeve,
            "target_ann_vol": vt_target_ann_vol_from_stack(raw),
            "vt_star": raw.get("VT_STAR") or raw.get("VOL_STAR"),
            "pick": raw.get("VT_TARGET_PICK_STAR"),
            "written_at": raw.get("VT_TARGET_WRITTEN_AT_STAR"),
            "star_stack_path": loc,
        }
    if "target_ann_vol" in raw:
        return raw
    return {
        "sleeve": sleeve,
        "target_ann_vol": vt_target_ann_vol_from_stack(raw),
        "vt_star": raw.get("VT_STAR") or raw.get("VOL_STAR"),
        "star_stack_path": loc,
    }


def write_leverage_artifact(*args, **kwargs) -> str:
    raise RuntimeError(
        "write_leverage_artifact is retired; persist VT_TARGET_ANN_VOL_STAR "
        "via backtest.star_stack_io.update_star_stack_key / run_leverage_policy(star_stack_path=...)."
    )
