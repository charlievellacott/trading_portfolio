"""Tiny JSON artifacts so MC / prop-firm notebooks can default to the desk pick."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from risk.monte_carlo.loaders import find_repo_root


def artifact_path(repo_root: str, sleeve: str) -> str:
    return os.path.join(
        repo_root, "06_risk", "leverage", "artifacts", f"{sleeve}_leverage.json"
    )


def write_leverage_artifact(
    path: str,
    *,
    sleeve: str,
    target_ann_vol: float | None,
    vt_star: str | None = None,
    vt_fields: dict | None = None,
    is_end: str | None = None,
    max_oos_dd: float | None = None,
    half_kelly_vol: float | None = None,
    pick: str = "calmar",
    extra: dict | None = None,
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: dict = {
        "sleeve": sleeve,
        "target_ann_vol": target_ann_vol,
        "vt_star": vt_star,
        "vt_fields": vt_fields or {},
        "is_end": is_end,
        "max_oos_dd": max_oos_dd,
        "half_kelly_vol": half_kelly_vol,
        "pick": pick,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load_leverage_artifact(
    repo_root: str | None = None,
    sleeve: str = "s1",
    *,
    path: str | None = None,
) -> dict | None:
    """Return the JSON dict, or ``None`` if the file is missing."""
    root = find_repo_root(repo_root) if path is None else repo_root
    loc = path if path is not None else artifact_path(str(root), sleeve)
    if not os.path.isfile(loc):
        return None
    with open(loc, encoding="utf-8") as f:
        return json.load(f)
