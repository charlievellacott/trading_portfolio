"""Offline unit tests for S3 metrics + runner gross Sharpe wiring."""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s3_fx_trend import runner as s3_runner
from strategies.s3_fx_trend.metrics import (
    corr_to_s2,
    metrics_from_returns,
)


def _toy_returns(n: int = 252, mu: float = 0.001, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    r = rng.normal(mu, 0.01, size=n)
    return pd.Series(r, index=idx, name="ret")


def test_calmar_positive_on_upward_path():
    idx = pd.date_range("2018-01-01", periods=252, freq="B")
    # Mostly positive path with a shallow drawdown so Calmar is defined.
    r = pd.Series(0.002, index=idx)
    r.iloc[100:105] = -0.01
    m = metrics_from_returns(r)
    assert m["n_days"] == 252
    assert np.isfinite(m["ann_sharpe"])
    assert m["ann_sharpe"] > 0
    assert m["max_drawdown"] < 0
    assert np.isfinite(m["calmar"])
    assert m["calmar"] > 0


def test_corr_to_s2_sanity():
    a = _toy_returns(120, mu=0.0, seed=1)
    # Perfect correlation
    assert corr_to_s2(a, a) == pytest.approx(1.0)
    # Anti-correlated
    assert corr_to_s2(a, -a) == pytest.approx(-1.0)
    # Sparse / empty → NaN
    assert np.isnan(corr_to_s2(a, pd.Series(dtype=float)))
    short = a.iloc[:2]
    assert np.isnan(corr_to_s2(short, short))


def test_ann_sharpe_gross_present_in_runner():
    src = inspect.getsource(s3_runner.run_s3_backtest)
    assert "ann_sharpe_gross" in src
    assert "returns_gross" in src
