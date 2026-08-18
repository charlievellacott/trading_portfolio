"""S2 walk-forward folds: expanding, embargo, no val dates in train."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.walkforward import embargo_bars_for_config, make_s2_folds


def test_three_folds_embargo_disjoint():
    idx = pd.date_range("2015-01-01", periods=400, freq="B")
    folds = make_s2_folds(idx, n_folds=3, embargo_bars=5)
    assert len(folds) == 3
    for f in folds:
        assert f.val_dates.min() > f.train_dates.max()
        overlap = set(f.train_dates).intersection(f.val_dates)
        assert not overlap
        assert set(f.embargo_dates).isdisjoint(f.val_dates)
        assert set(f.embargo_dates).isdisjoint(f.train_dates)


def test_embargo_bars_1h_vs_1d():
    assert embargo_bars_for_config(bar="1d") == 5
    assert embargo_bars_for_config(bar="1h") == 30


def test_too_short_raises():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    with pytest.raises(ValueError, match="too short"):
        make_s2_folds(idx, n_folds=3, embargo_bars=5)
