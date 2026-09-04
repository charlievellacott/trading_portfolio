"""S1 star-stack path and I/O re-exports. No sim-config mapper here."""

from __future__ import annotations

import os

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

__all__ = [
    "ARTIFACTS_DIR",
    "DEFAULT_STAR_STACK",
    "load_star_stack",
    "require_star",
    "save_star_stack",
    "update_star_stack_key",
    "vt_target_ann_vol_from_stack",
]
