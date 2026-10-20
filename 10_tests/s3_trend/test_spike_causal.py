"""Spike delever must not scale the bar that revealed the spike."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.s3_trend.engine import _spike_scale


def test_spike_scale_applies_next_bar():
    idx = pd.bdate_range("2020-01-02", periods=40)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, len(idx)), index=idx)
    r.iloc[30] = 0.50
    scale = _spike_scale(r, threshold_sigmas=3.0, window=20)
    assert float(scale.iloc[30]) == 1.0
    assert float(scale.iloc[31]) == 0.5
    assert np.isfinite(scale).all()
