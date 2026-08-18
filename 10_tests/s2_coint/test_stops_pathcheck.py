"""H-008 path-check: adverse H/L after entry can flatten; 1% × scale notional."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.engine import simulate_pair
from strategies.s2_coint.sizing import atr_size_multiplier
from strategies.s2_coint.spread_ohlc import spread_ohlc_frame


def test_atr_size_multiplier_scales_with_pair_scale():
    m1 = atr_size_multiplier(atr=0.02, beta=1.0, n_pairs=3, pair_scale=1.0, leverage=1.0)
    m2 = atr_size_multiplier(atr=0.02, beta=1.0, n_pairs=3, pair_scale=2.0, leverage=1.0)
    assert m2 == pytest.approx(2.0 * m1)


def test_hl3_atr_breaker_exits_on_adverse_spread_low():
    n = 20
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    z = np.zeros(n)
    z[0] = -2.5
    px = np.full(n, 10.0)
    panel = pd.DataFrame(
        {
            "date": idx,
            "pair_id": "1398.HK|0939.HK",
            "ticker_y": "1398.HK",
            "ticker_x": "0939.HK",
            "open_y": px,
            "high_y": px,
            "low_y": px,
            "close_y": px,
            "open_x": px * 0.5,
            "high_x": px * 0.5,
            "low_x": px * 0.5,
            "close_x": px * 0.5,
            "alpha": 0.0,
            "beta": 1.0,
            "spread": 0.0,
            "z": z,
            "half_life": 2.0,
            "adf_pvalue": 0.01,
        }
    )
    ohlc = spread_ohlc_frame(panel)
    panel["spread_open"] = ohlc["spread_open"].to_numpy()
    panel["spread_high"] = ohlc["spread_high"].to_numpy()
    panel["spread_low"] = ohlc["spread_low"].to_numpy()
    panel["spread_close"] = ohlc["spread_close"].to_numpy()
    panel["atr_spread"] = 0.05
    # Crash the fill-bar spread low to force a stop.
    panel.loc[panel.index[1], "spread_low"] = -10.0
    res = simulate_pair(panel, S2SimConfig(exit_mode="hl3_atr_breaker"))
    assert res.n_entries >= 1
    assert res.n_open_at_end == 0 or len(res.trades) >= 1
