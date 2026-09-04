"""Desk helper: surface + half-Kelly policy + star-stack vol-target write."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from backtest.star_stack_io import VT_TARGET_ANN_VOL_STAR, update_star_stack_key
from risk.analytics.leverage.apply import cfg_with_target, overlay_vol_target
from risk.analytics.leverage.policy import apply_policy, half_kelly_target_vol
from risk.analytics.leverage.surfaces import leverage_surface
from risk.analytics.s1_equities.vol_targeting import VolTargetConfig, vol_target_star


def compute_leverage_surface(
    base: pd.Series,
    cfg: VolTargetConfig,
    *,
    targets: list[float],
    is_end: str | pd.Timestamp,
    periods_per_year: float,
    max_oos_dd: float = 0.25,
    pick: str = "calmar",
) -> dict:
    """Grid ``target_ann_vol`` and return policy suggestion only (no star-stack write)."""
    surface = leverage_surface(
        base,
        cfg,
        targets=list(targets),
        is_end=is_end,
        periods_per_year=float(periods_per_year),
    )
    hk = half_kelly_target_vol(base, periods_per_year=float(periods_per_year))
    suggestion = apply_policy(
        surface,
        half_kelly_vol=hk,
        max_oos_dd=float(max_oos_dd),
        pick=pick,
    )
    return {
        "surface": surface,
        "half_kelly_vol": hk,
        "suggestion": suggestion,
        "pick": pick,
        "max_oos_dd": float(max_oos_dd),
    }


def freeze_vt_target_ann_vol(
    star_stack_path: str,
    target_ann_vol: float,
    *,
    pick: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Persist desk ``VT_TARGET_ANN_VOL_STAR`` via ``backtest.star_stack_io``."""
    meta = {
        "VT_TARGET_WRITTEN_AT_STAR": datetime.now(timezone.utc).isoformat(),
        "VT_TARGET_SOURCE_STAR": "manual",
    }
    if pick is not None:
        meta["VT_TARGET_PICK_STAR"] = pick
    if extra:
        meta.update(extra)
    return update_star_stack_key(
        star_stack_path,
        VT_TARGET_ANN_VOL_STAR,
        float(target_ann_vol),
        extra=meta,
    )


def run_leverage_policy(
    base: pd.Series,
    cfg: VolTargetConfig,
    *,
    targets: list[float],
    is_end: str | pd.Timestamp,
    periods_per_year: float,
    max_oos_dd: float = 0.25,
    pick: str = "calmar",
    star_stack_path: str | None = None,
    sleeve: str = "s1",
    vt_star: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Sweep ``target_ann_vol``; optionally auto-persist policy pick to star stack."""
    pack = compute_leverage_surface(
        base,
        cfg,
        targets=list(targets),
        is_end=is_end,
        periods_per_year=float(periods_per_year),
        max_oos_dd=float(max_oos_dd),
        pick=pick,
    )
    decision = pack["suggestion"]
    star = vt_star if vt_star is not None else (vol_target_star(cfg) if cfg.enabled else "none")
    path = None
    if star_stack_path and decision.get("target_ann_vol") is not None:
        freeze_vt_target_ann_vol(
            star_stack_path,
            float(decision["target_ann_vol"]),
            pick=pick,
            extra=extra,
        )
        path = star_stack_path
    chosen = None
    if decision.get("target_ann_vol") is not None:
        chosen = overlay_vol_target(
            base, cfg_with_target(cfg, float(decision["target_ann_vol"]))
        )
    return {
        "surface": pack["surface"],
        "half_kelly_vol": pack["half_kelly_vol"],
        "decision": decision,
        "suggestion": decision,
        "star_stack_path": path,
        "artifact_path": path,
        "chosen_returns": chosen,
        "vt_star": star,
        "sleeve": sleeve,
    }
