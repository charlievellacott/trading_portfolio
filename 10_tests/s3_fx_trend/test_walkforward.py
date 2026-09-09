"""Offline unit tests for S3 walk-forward folds + embargo."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s3_fx_trend.walkforward import embargo_bars_for_config, make_s3_folds


def test_make_s3_folds_length_and_embargo():
    dates = pd.date_range("2010-01-01", periods=200, freq="B")
    folds = make_s3_folds(dates, n_folds=3, embargo_bars=5)
    assert len(folds) == 3
    for f in folds:
        assert len(f.embargo_dates) == 5
        assert len(f.train_dates) > 0
        assert len(f.val_dates) > 0
        assert f.train_dates.max() < f.embargo_dates.min()
        assert f.embargo_dates.max() < f.val_dates.min()
        # Expanding: later folds have longer train pools (post-embargo).
    assert len(folds[2].train_dates) > len(folds[0].train_dates)
    # No train/val overlap
    for f in folds:
        overlap = set(f.train_dates) & set(f.val_dates)
        assert not overlap
        assert set(f.embargo_dates).isdisjoint(set(f.train_dates))
        assert set(f.embargo_dates).isdisjoint(set(f.val_dates))


def test_embargo_bars_for_config():
    assert embargo_bars_for_config(bar="1d") == 5
    assert embargo_bars_for_config(bar="1h") == 30


def test_make_s3_folds_too_short_raises():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    with pytest.raises(ValueError, match="too short"):
        make_s3_folds(dates, n_folds=3, embargo_bars=5)
