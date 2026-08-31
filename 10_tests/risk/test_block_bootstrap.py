"""Tests for stationary block bootstrap path generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk.analytics.monte_carlo.block_bootstrap import (
    StationaryBlockBootstrap,
    split_joint_simulations,
    stationary_bootstrap_indices,
)
from risk.analytics.monte_carlo.hmm_simulator import GaussianHMMSimulator


def test_indices_in_range_no_negative_lags():
    rng = np.random.default_rng(0)
    n_obs = 17
    idx = stationary_bootstrap_indices(n_obs, 40, 25, 5.0, rng)
    assert idx.shape == (40, 25)
    assert int(idx.min()) >= 0
    assert int(idx.max()) < n_obs


def test_joint_paths_share_block_indices():
    n = 30
    strat = pd.Series(np.arange(n, dtype=float), name="strategy")
    spy = pd.Series(100.0 * np.arange(n, dtype=float), name="spy")
    frame = pd.concat([strat, spy], axis=1)
    sim = StationaryBlockBootstrap(
        n_simulations=8, horizon=20, random_seed=3, mean_block_length=4.0
    )
    sim.fit(frame)
    paths = sim.simulate(20)
    s, b = split_joint_simulations(paths)
    np.testing.assert_allclose(s.to_numpy() * 100.0, b.to_numpy())


def test_univariate_shape_and_summary():
    r = pd.Series(np.linspace(-0.01, 0.02, 50), name="strategy")
    sim = StationaryBlockBootstrap(
        n_simulations=12, random_seed=1, mean_block_length=6.0
    )
    sim.fit(r)
    paths = sim.simulate(10)
    assert paths.shape == (10, 12)
    assert list(paths.columns) == [f"sim_{i}" for i in range(12)]
    summ = sim.summary(paths)
    assert "mean_wealth" in summ.columns
    assert summ.shape[0] == 1


def test_simulate_requires_fit():
    sim = StationaryBlockBootstrap(2, mean_block_length=3.0)
    with pytest.raises(RuntimeError, match="fit"):
        sim.simulate(5)


def test_hmm_univariate_rejects_joint_frame():
    frame = pd.DataFrame({"strategy": [0.01] * 30, "spy": [0.0] * 30})
    hmm = GaussianHMMSimulator(n_simulations=4, random_seed=0)
    with pytest.raises(TypeError, match="univariate"):
        hmm.fit(frame)


def test_hmm_simulate_shape():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, 80))
    hmm = GaussianHMMSimulator(n_simulations=6, random_seed=2)
    hmm.fit(r)
    paths = hmm.simulate(15)
    assert paths.shape == (15, 6)
    assert hmm.params is not None
