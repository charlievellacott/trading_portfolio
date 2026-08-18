"""Kalman correlation reuses _run_kalman; PIT prior (burn-in NaN)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.processing.feature_implementation.kalman import _run_kalman, kalman_correlation


def test_kalman_correlation_burn_in_and_bounds():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    x = pd.Series(rng.normal(size=200), index=idx)
    y = x + 0.1 * pd.Series(rng.normal(size=200), index=idx)
    rho = kalman_correlation(x, y, burn_in=30)
    assert rho.iloc[:30].isna().all()
    finite = rho.dropna()
    assert finite.between(-1.0, 1.0).all()
    assert float(finite.iloc[-1]) > 0.3


def test_run_kalman_still_exported_for_correlation():
    n = 10
    y = np.ones(n)
    F = np.ones((n, 1))
    Q = np.array([[1e-4]])
    prior, _, _, _, _ = _run_kalman(y, F, Q, 1e-3, np.zeros(1), np.eye(1))
    assert prior.shape == (n, 1)
