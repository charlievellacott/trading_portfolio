"""Spread OHLC envelope uses close-t β; H/L are not hedge inputs."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.s2_coint.spread_ohlc import attach_spread_indicators, spread_ohlc_frame


def test_spread_high_uses_y_high_and_x_low_when_beta_positive():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    g = pd.DataFrame(
        {
            "date": idx,
            "alpha": 0.0,
            "beta": 1.0,
            "open_y": 10.0,
            "high_y": 12.0,
            "low_y": 9.0,
            "close_y": 10.0,
            "open_x": 5.0,
            "high_x": 6.0,
            "low_x": 4.0,
            "close_x": 5.0,
        }
    )
    ohlc = spread_ohlc_frame(g)
    # log(12) - log(4) > log(10) - log(5)
    assert float(ohlc["spread_high"].iloc[0]) > float(ohlc["spread_close"].iloc[0])
    assert float(ohlc["spread_low"].iloc[0]) < float(ohlc["spread_close"].iloc[0])


def test_attach_atr_without_talib():
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    px = np.linspace(10.0, 12.0, 20)
    panel = pd.DataFrame(
        {
            "date": idx,
            "pair_id": "1398.HK|0939.HK",
            "alpha": 0.0,
            "beta": 1.0,
            "open_y": px,
            "high_y": px * 1.01,
            "low_y": px * 0.99,
            "close_y": px,
            "open_x": px * 0.5,
            "high_x": px * 0.51,
            "low_x": px * 0.49,
            "close_x": px * 0.5,
        }
    )
    out = attach_spread_indicators(panel, include_rsi_adx=False)
    assert "atr_spread" in out.columns
    assert "rsi_spread" not in out.columns


def test_attach_adds_rsi_adx_atr_columns():
    talib = pytest.importorskip("talib")
    _ = talib
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    px = np.linspace(10.0, 12.0, 40)
    panel = pd.DataFrame(
        {
            "date": idx,
            "pair_id": "1398.HK|0939.HK",
            "alpha": 0.0,
            "beta": 1.0,
            "open_y": px,
            "high_y": px * 1.01,
            "low_y": px * 0.99,
            "close_y": px,
            "open_x": px * 0.5,
            "high_x": px * 0.51,
            "low_x": px * 0.49,
            "close_x": px * 0.5,
        }
    )
    out = attach_spread_indicators(panel)
    for col in ("spread_open", "spread_high", "spread_low", "spread_close", "rsi_spread", "adx_spread", "atr_spread"):
        assert col in out.columns
