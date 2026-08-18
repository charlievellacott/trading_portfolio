"""S2 research helpers: lookbacks, STAR config, corr gate."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.research import (
    BARS_PER_SESSION_1H,
    config_from_stack,
    lookbacks_for_bar,
    unique_tickers,
)
from strategies.s2_coint.overlap import corr_blocks_candidate


def test_lookbacks_1h_are_session_scaled_not_raw_hours():
    d = lookbacks_for_bar("1d")
    h = lookbacks_for_bar("1h")
    assert d["ols_window"] == 252
    assert h["ols_window"] == 252 * BARS_PER_SESSION_1H
    assert h["z_window"] == 60 * BARS_PER_SESSION_1H
    with pytest.raises(ValueError, match="4h"):
        lookbacks_for_bar("4h")


def test_config_from_stack_defaults_and_override():
    stack = {
        "HEDGE_STAR": "ols",
        "BAR_STAR": "1d",
        "BREAK_STAR": None,
        "HL_GATE_STAR": "5_60",
        "CORR_GATE_STAR": "off",
    }
    cfg = config_from_stack(stack, break_mode="flat_05")
    assert cfg.bar == "1d"
    assert cfg.break_mode == "flat_05"
    assert cfg.hl_gate_min == 5.0
    assert cfg.hl_gate_max == 60.0
    assert cfg.corr_k is None


def test_unique_tickers_locked_c():
    ids = ["1398.HK|0939.HK", "1288.HK|3328.HK", "8306.T|8316.T"]
    t = unique_tickers(ids)
    assert t == ["1398.HK", "0939.HK", "1288.HK", "3328.HK", "8306.T", "8316.T"]


def test_corr_blocks_open_pair_and_same_bar_priority():
    rho = {tuple(sorted(("A", "B"))): 0.9}
    assert corr_blocks_candidate(
        "B",
        1.0,
        open_ids=["A"],
        same_bar_candidates=[],
        abs_rho=rho,
        k=0.5,
    )
    assert not corr_blocks_candidate(
        "B",
        2.0,
        open_ids=[],
        same_bar_candidates=[("A", 1.0), ("B", 2.0)],
        abs_rho=rho,
        k=0.5,
    )
    assert corr_blocks_candidate(
        "B",
        1.0,
        open_ids=[],
        same_bar_candidates=[("A", 2.0), ("B", 1.0)],
        abs_rho=rho,
        k=0.5,
    )
