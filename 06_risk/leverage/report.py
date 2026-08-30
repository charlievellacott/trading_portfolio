"""Desk helper: surface + half-Kelly policy + artifact write."""

from __future__ import annotations

import pandas as pd

from risk.leverage.apply import cfg_with_target, overlay_vol_target
from risk.leverage.artifacts import write_leverage_artifact
from risk.leverage.policy import apply_policy, half_kelly_target_vol
from risk.leverage.surfaces import leverage_surface
from risk.s1_equities.vol_targeting import VolTargetConfig, vol_target_star


def run_leverage_policy(
    base: pd.Series,
    cfg: VolTargetConfig,
    *,
    targets: list[float],
    is_end: str | pd.Timestamp,
    periods_per_year: float,
    max_oos_dd: float = 0.25,
    pick: str = "calmar",
    artifact_path: str | None = None,
    sleeve: str = "s1",
    vt_star: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Sweep ``target_ann_vol``, apply veto/ceiling, optionally persist JSON."""
    surface = leverage_surface(
        base,
        cfg,
        targets=list(targets),
        is_end=is_end,
        periods_per_year=float(periods_per_year),
    )
    hk = half_kelly_target_vol(base, periods_per_year=float(periods_per_year))
    decision = apply_policy(
        surface,
        half_kelly_vol=hk,
        max_oos_dd=float(max_oos_dd),
        pick=pick,
    )
    star = vt_star if vt_star is not None else (vol_target_star(cfg) if cfg.enabled else "none")
    path = None
    if artifact_path:
        path = write_leverage_artifact(
            artifact_path,
            sleeve=sleeve,
            target_ann_vol=decision.get("target_ann_vol"),
            vt_star=star,
            vt_fields={
                "estimator": cfg.estimator,
                "target_ann_vol_frozen_family": True,
                "periods_per_year": float(cfg.periods_per_year),
                "min_periods": int(cfg.min_periods),
                "forget": float(cfg.forget),
                "quantile": float(cfg.quantile),
                "deadband": float(cfg.deadband),
            },
            is_end=str(pd.Timestamp(is_end).date()),
            max_oos_dd=float(max_oos_dd),
            half_kelly_vol=float(hk),
            pick=pick,
            extra=extra,
        )
    chosen = None
    if decision.get("target_ann_vol") is not None:
        chosen = overlay_vol_target(
            base, cfg_with_target(cfg, float(decision["target_ann_vol"]))
        )
    return {
        "surface": surface,
        "half_kelly_vol": hk,
        "decision": decision,
        "artifact_path": path,
        "chosen_returns": chosen,
    }
