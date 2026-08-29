"""Tests for EV significance (HAC + bootstrap) and vs-SPY metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk.monte_carlo.block_bootstrap import split_joint_simulations
from risk.monte_carlo.ev_stats import (
    cvar,
    ev_significance,
    excess_returns,
    hac_mean_inference,
    horizon_ev,
    p_not_beat_spy,
    scale_simple_returns,
    simulate_joint_paths,
    underwater_probs,
)


def test_hac_constant_positive_mean_excludes_zero():
    r = pd.Series(np.full(80, 0.01))
    hac = hac_mean_inference(r, lags=0, periods_per_year=252.0)
    assert abs(hac["mean"] - 0.01) < 1e-12
    assert hac["t_stat"] > 10.0
    assert hac["p_value"] < 1e-6
    assert hac["ci_excludes_zero"] is True
    assert hac["ci_low"] > 0.0


def test_hac_zero_mean_ci_includes_zero():
    r = pd.Series(np.zeros(60))
    hac = hac_mean_inference(r, lags=0)
    assert hac["mean"] == 0.0
    assert hac["ci_excludes_zero"] is False


def test_ev_significance_bootstrap_p_small_when_mean_positive():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.004, 0.008, 200))
    sig = ev_significance(
        r,
        periods_per_year=52.0,
        mean_block_length=5.0,
        n_bootstrap=400,
        random_seed=11,
    )
    assert sig["mean"] > 0.0
    assert sig["bootstrap_p_mean_le_0"] < 0.05
    assert "t_stat" in sig.index
    assert "p_value" in sig.index
    assert "psr" in sig.index


def test_p_not_beat_spy_paired_terminals():
    strat = pd.DataFrame(np.full((5, 7), 0.02))
    spy = pd.DataFrame(np.zeros((5, 7)))
    assert p_not_beat_spy(strat, spy) == 0.0
    assert p_not_beat_spy(spy, strat) == 1.0


def test_excess_hac_distinct_from_p_not_beat():
    idx = pd.bdate_range("2020-01-01", periods=40)
    strat = pd.Series(0.01, index=idx, name="strategy")
    spy = pd.Series(0.001, index=idx, name="spy")
    excess = excess_returns(strat, spy)
    hac = hac_mean_inference(excess, lags=0)
    assert abs(hac["mean"] - 0.009) < 1e-12
    assert hac["ci_excludes_zero"] is True
    paths_s = pd.DataFrame(np.full((10, 20), 0.01))
    paths_b = pd.DataFrame(np.full((10, 20), 0.001))
    assert p_not_beat_spy(paths_s, paths_b) == 0.0


def test_horizon_ev_and_cvar_underwater():
    paths = pd.DataFrame(np.full((4, 11), -0.02))
    hev = horizon_ev(paths)
    assert hev["mean_terminal"] < 0.0
    term = (1.0 + paths).prod(axis=0) - 1.0
    assert cvar(term, alpha=0.2) <= float(np.quantile(term, 0.2)) + 1e-12
    uw = underwater_probs(paths)
    assert uw["p_terminal_underwater"] == 1.0
    assert uw["p_ever_underwater"] == 1.0


def test_scale_simple_returns_leaves_spy():
    d = pd.DataFrame({"strategy": [0.02, -0.01], "spy": [0.01, 0.01]})
    out = scale_simple_returns(d, 2.0)
    assert list(out["strategy"]) == [0.04, -0.02]
    assert list(out["spy"]) == [0.01, 0.01]


def test_joint_simulate_preserves_pairing():
    n = 25
    frame = pd.DataFrame(
        {"strategy": np.arange(n, dtype=float), "spy": 3.0 * np.arange(n, dtype=float)}
    )
    joint = simulate_joint_paths(
        frame, n_simulations=5, horizon=12, mean_block_length=3.0, random_seed=4
    )
    s, b = split_joint_simulations(joint)
    np.testing.assert_allclose(s.to_numpy() * 3.0, b.to_numpy())
