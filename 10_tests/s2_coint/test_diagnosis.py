"""Unit tests for Asia C failure diagnosis helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.diagnosis import (
    enrich_trades,
    extreme_trades,
    slice_panel_window,
)


def _mini_panel(*, n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    z = np.zeros(n, dtype=float)
    z[0] = -2.5
    z[1] = -2.0
    z[3] = 0.5
    px = np.linspace(10.0, 11.0, n)
    return pd.DataFrame(
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
            "beta": 2.0,
            "spread": np.linspace(-0.1, 0.1, n),
            "z": z,
            "half_life": 20.0,
            "adf_pvalue": 0.01,
        }
    )


def test_slice_panel_window_is_only_drops_oos():
    panel = _mini_panel(n=8)
    # Force two dates after IS end.
    panel.loc[panel.index[-2:], "date"] = pd.to_datetime(
        ["2022-01-03", "2022-01-04"]
    )
    is_end = "2021-12-31"
    is_only = slice_panel_window(panel, is_end, use_oos=False)
    full = slice_panel_window(panel, is_end, use_oos=True)
    assert len(full) == len(panel)
    assert is_only["date"].max() <= pd.Timestamp(is_end)
    assert len(is_only) == len(panel) - 2


def test_enrich_trades_pnl_pct_compounds_inclusive():
    panel = _mini_panel(n=8)
    # Fill dates: entry on bar 1, exit on bar 3 (0-indexed dates).
    entry = pd.Timestamp(panel["date"].iloc[1])
    exit_ = pd.Timestamp(panel["date"].iloc[3])
    trades = pd.DataFrame(
        [
            {
                "pair_id": "1398.HK|0939.HK",
                "side": 1,
                "entry_date": entry,
                "exit_date": exit_,
                "hold_bars": 2,
                "entry_cost_bps": 10.0,
                "exit_cost_bps": 10.0,
            }
        ]
    )
    # Known daily net returns on fill calendar.
    rets = pd.Series(
        {
            entry: 0.01,
            pd.Timestamp(panel["date"].iloc[2]): 0.02,
            exit_: -0.005,
        },
        dtype=float,
    )
    enriched = enrich_trades(trades, panel, rets)
    assert len(enriched) == 1
    expected = ((1.01 * 1.02 * 0.995) - 1.0) * 100.0
    assert enriched.iloc[0]["pnl_pct"] == pytest.approx(expected)
    # Signal bar is the row before fill entry.
    assert enriched.iloc[0]["signal_date"] == pd.Timestamp(panel["date"].iloc[0])
    assert enriched.iloc[0]["z_entry"] == pytest.approx(-2.5)
    assert enriched.iloc[0]["side_label"] == "long"


def test_extreme_trades_ranks_by_pnl_pct():
    trades = pd.DataFrame(
        [
            {
                "pair_id": "a|b",
                "side": 1,
                "side_label": "long",
                "entry_date": "2018-01-02",
                "exit_date": "2018-01-03",
                "hold_bars": 1,
                "entry_cost_bps": 1.0,
                "exit_cost_bps": 1.0,
                "signal_date": "2018-01-01",
                "z_entry": -2.0,
                "spread_entry": 0.0,
                "pnl_pct": 1.0,
            },
            {
                "pair_id": "a|b",
                "side": -1,
                "side_label": "short",
                "entry_date": "2018-02-02",
                "exit_date": "2018-02-03",
                "hold_bars": 1,
                "entry_cost_bps": 1.0,
                "exit_cost_bps": 1.0,
                "signal_date": "2018-02-01",
                "z_entry": 2.0,
                "spread_entry": 0.0,
                "pnl_pct": -3.0,
            },
            {
                "pair_id": "a|b",
                "side": 1,
                "side_label": "long",
                "entry_date": "2018-03-02",
                "exit_date": "2018-03-03",
                "hold_bars": 1,
                "entry_cost_bps": 1.0,
                "exit_cost_bps": 1.0,
                "signal_date": "2018-03-01",
                "z_entry": -2.1,
                "spread_entry": 0.0,
                "pnl_pct": 5.0,
            },
        ]
    )
    best, worst = extreme_trades(trades, n=1)
    assert float(best.iloc[0]["pnl_pct"]) == pytest.approx(5.0)
    assert float(worst.iloc[0]["pnl_pct"]) == pytest.approx(-3.0)
