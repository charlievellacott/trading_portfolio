"""Tests for PSR / DSR inference helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from performance.sharpe_inference import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    return_moments,
)


def test_return_moments_empty():
    m = return_moments(pd.Series(dtype=float))
    assert m["n_obs"] == 0
    assert np.isnan(m["sr"])


def test_return_moments_short_series():
    m = return_moments(pd.Series([0.01, 0.02, 0.01]))
    assert m["n_obs"] == 3
    assert np.isnan(m["sr"])


def test_psr_increases_with_higher_sharpe():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.001, 0.01, 300))
    m = return_moments(r)
    low = probabilistic_sharpe_ratio(m["sr"] * 0.5, m["n_obs"], m["skew"], m["kurtosis"])
    high = probabilistic_sharpe_ratio(m["sr"], m["n_obs"], m["skew"], m["kurtosis"])
    assert high > low


def test_dsr_decreases_with_more_trials():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0012, 0.01, 400))
    m = return_moments(r)
    d1 = deflated_sharpe_ratio(m["sr"], m["n_obs"], m["skew"], m["kurtosis"], 1)
    d20 = deflated_sharpe_ratio(m["sr"], m["n_obs"], m["skew"], m["kurtosis"], 20)
    assert np.isfinite(d1) and np.isfinite(d20)
    assert d1 > d20


def test_psr_benchmark_one():
    rng = np.random.default_rng(99)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    m = return_moments(r)
    psr = probabilistic_sharpe_ratio(
        m["sr"], m["n_obs"], m["skew"], m["kurtosis"], sr_benchmark=1.0
    )
    assert 0.0 <= psr <= 1.0 or np.isnan(psr)
