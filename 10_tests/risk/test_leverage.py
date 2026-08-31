"""Vol-target overlay, half-Kelly ceiling, and drawdown veto."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd

from risk.analytics.leverage.apply import overlay_vol_target
from risk.analytics.leverage.loaders import require_base_parquet
from risk.analytics.leverage.policy import apply_policy, half_kelly_target_vol
from risk.analytics.leverage.report import run_leverage_policy
from risk.analytics.s1_equities.vol_targeting import ESTIMATOR_ROLLING, VolTargetConfig


def _quiet_weekly(n: int = 160, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n, freq="W-MON")
    return pd.Series(rng.normal(0.002, 0.012, n), index=idx, name="ret")


def test_higher_target_vol_higher_realized_vol():
    r = _quiet_weekly()
    cfg = VolTargetConfig(
        estimator=ESTIMATOR_ROLLING,
        window=20,
        min_periods=10,
        deadband=0.0,
        max_leverage=8.0,
        min_leverage=0.05,
        periods_per_year=52.0,
    )
    vols = []
    for t in (0.06, 0.12, 0.18):
        out = overlay_vol_target(r, replace(cfg, target_ann_vol=t))
        vols.append(float(out.std(ddof=1) * np.sqrt(52.0)))
    assert vols[0] < vols[1] < vols[2]


def test_dd_veto_not_selected():
    surface = pd.DataFrame(
        {
            "target_ann_vol": [0.06, 0.10, 0.18],
            "oos_calmar": [0.40, 0.90, 1.50],
            "oos_cagr": [0.05, 0.08, 0.12],
            "oos_max_drawdown": [-0.10, -0.20, -0.40],
            "oos_sharpe": [1.0, 1.0, 1.0],
        }
    )
    dec = apply_policy(surface, half_kelly_vol=0.20, max_oos_dd=0.25, pick="calmar")
    assert dec["target_ann_vol"] == 0.10
    assert dec["reason"] == "ok"


def test_half_kelly_ceiling_respected():
    surface = pd.DataFrame(
        {
            "target_ann_vol": [0.06, 0.10, 0.18],
            "oos_calmar": [0.20, 0.90, 1.50],
            "oos_cagr": [0.04, 0.08, 0.12],
            "oos_max_drawdown": [-0.08, -0.12, -0.15],
            "oos_sharpe": [1.0, 1.0, 1.0],
        }
    )
    dec = apply_policy(surface, half_kelly_vol=0.07, max_oos_dd=0.50, pick="calmar")
    assert dec["target_ann_vol"] == 0.06
    assert float(dec["n_survivors"]) == 1


def test_half_kelly_positive_mean():
    r = pd.Series(np.full(80, 0.01))
    hk = half_kelly_target_vol(r, periods_per_year=52.0)
    assert hk > 0.0


def test_run_leverage_policy_writes_artifact(tmp_path):
    r = _quiet_weekly(120, seed=3)
    cfg = VolTargetConfig(
        estimator=ESTIMATOR_ROLLING,
        window=16,
        min_periods=8,
        deadband=0.0,
        max_leverage=4.0,
        target_ann_vol=0.10,
        periods_per_year=52.0,
    )
    out = os.path.join(str(tmp_path), "s1_leverage.json")
    pack = run_leverage_policy(
        r,
        cfg,
        targets=[0.06, 0.10, 0.14],
        is_end="2018-01-01",
        periods_per_year=52.0,
        max_oos_dd=0.80,
        pick="calmar",
        artifact_path=out,
        sleeve="s1",
        vt_star="vt_rolling_16_10_q0.5",
    )
    assert pack["surface"].shape[0] == 3
    assert os.path.isfile(out)
    assert "half_kelly_vol" in pack


def test_missing_base_parquet_is_loud(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    try:
        require_base_parquet(missing, "export hint")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "Unlevered base period returns missing" in str(exc)
        assert "export hint" in str(exc)
