"""Apply the live VT overlay to an unlevered base return series."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from risk.analytics.s1_equities.vol_targeting import (
    VolTargetConfig,
    leverage_series,
    parse_vol_target_star,
)


def cfg_with_target(cfg: VolTargetConfig, target_ann_vol: float) -> VolTargetConfig:
    """Copy ``cfg`` with only ``target_ann_vol`` changed."""
    return replace(cfg, target_ann_vol=float(target_ann_vol))


def s1_frozen_cfg(
    vt_star: str,
    *,
    periods_per_year: float = 52.0,
) -> VolTargetConfig:
    """Parse ``VT_STAR`` and pin annualization to S1 weeks."""
    cfg = parse_vol_target_star(vt_star)
    return replace(cfg, periods_per_year=float(periods_per_year))


def s2_frozen_cfg(
    *,
    target_ann_vol: float = 0.10,
    periods_per_year: float = 252.0,
    sigma_window: int = 60,
) -> VolTargetConfig:
    """S2 engine ``s1_vt`` family: Bayes defaults, ``min_periods`` from sigma window."""
    return VolTargetConfig(
        enabled=True,
        target_ann_vol=float(target_ann_vol),
        periods_per_year=float(periods_per_year),
        min_periods=max(13, int(sigma_window // 4)),
    )


def overlay_vol_target(base: pd.Series, cfg: VolTargetConfig) -> pd.Series:
    """Scale unlevered simple returns by PIT ``leverage_series`` (07 bakeoff overlay).

    This is the live VT API on **base** returns, not ``r' = k r`` on a sealed net parquet.
    Stops/costs scale with gross in the full runner; this overlay is the same
    approximation ``07_vol_target_bakeoff`` already uses.
    """
    r = pd.to_numeric(base, errors="coerce").astype(float)
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    lev = leverage_series(r, cfg)["leverage"].reindex(r.index).astype(float)
    out = (r * lev).rename("ret")
    return out
