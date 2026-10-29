"""Offline unit tests for S3 OANDA and Alpaca ETF cost models."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s3_trend.research import config_from_stack
from data.processing.s3_price_panel import ETF_MULTI_TICKERS
from strategies.s3_trend.config import S3SimConfig
from strategies.s3_trend.costs import (
    COSTS,
    DEFAULT_COST_PROFILE,
    ETF_COST_PROFILE,
    FX_COST_PROFILE,
    cost_profile_for_universe,
    daily_borrow_return,
    daily_swap_return,
    leg_cost_bps,
    pair_supported,
    resolve_cost_profile,
)
from strategies.s3_trend.engine import simulate_pair


LOCKED_SPREADS = {
    "EURUSD": 1.4,
    "USDJPY": 1.4,
    "GBPUSD": 2.0,
    "USDCHF": 1.8,
    "USDCAD": 2.2,
    "AUDUSD": 1.4,
    "NZDUSD": 1.7,
}


def test_spreads_match_locked_doc():
    cfg = COSTS[DEFAULT_COST_PROFILE]
    spreads = cfg["pair_spread_pips"]
    assert set(spreads) == set(LOCKED_SPREADS)
    for pair, pips in LOCKED_SPREADS.items():
        assert spreads[pair] == pytest.approx(pips)
    assert cfg["slippage_pips_per_leg"] == pytest.approx(0.1)
    assert cfg["swap"]["wednesday_triple_swap"] is True


def test_swap_wednesday_triple():
    wed = "2024-01-03"  # Wednesday
    thu = "2024-01-04"  # Thursday
    kwargs = dict(
        pair="EURUSD",
        weight=1.0,
        rate_diff=0.02,
        financing_spread_bps_annual=50.0,
    )
    r_wed = daily_swap_return(**kwargs, asof=wed, wednesday_triple=True)
    r_thu = daily_swap_return(**kwargs, asof=thu, wednesday_triple=True)
    assert r_wed == pytest.approx(3.0 * r_thu)
    # Disabled flag → no triple
    r_wed_off = daily_swap_return(**kwargs, asof=wed, wednesday_triple=False)
    assert r_wed_off == pytest.approx(r_thu)


def test_pair_supported_sek_false():
    assert pair_supported("EURUSD") is True
    assert pair_supported("USDSEK") is False
    assert pair_supported("EURSEK") is False
    assert pair_supported("SEK") is False


def test_etf_profile_locked_numbers():
    cfg = COSTS[ETF_COST_PROFILE]
    assert cfg["commission_bps"] == pytest.approx(0.0)
    assert cfg["third_party_fee_bps"] == pytest.approx(0.1)
    assert cfg["slippage_bps_per_leg"] == pytest.approx(3.2)
    assert cfg["borrow_bps_annual"] == pytest.approx(0.0)
    assert leg_cost_bps("SPY", 100.0, profile=ETF_COST_PROFILE) == pytest.approx(3.3)


def test_spy_unsupported_on_oanda_supported_on_etf():
    assert pair_supported("SPY") is False
    assert pair_supported("SPY", FX_COST_PROFILE) is False
    assert pair_supported("SPY", ETF_COST_PROFILE) is True
    assert pair_supported("EURUSD", ETF_COST_PROFILE) is False
    assert pair_supported("EURUSD", FX_COST_PROFILE) is True


def test_universe_routing():
    assert cost_profile_for_universe("A") == FX_COST_PROFILE
    assert cost_profile_for_universe("B") == ETF_COST_PROFILE
    assert cost_profile_for_universe(None) == FX_COST_PROFILE
    assert resolve_cost_profile("EURUSD", universe="A") == FX_COST_PROFILE
    assert resolve_cost_profile("SPY", universe="B") == ETF_COST_PROFILE
    assert resolve_cost_profile("SPY", override=FX_COST_PROFILE, universe="B") == FX_COST_PROFILE


def test_etb_borrow_is_zero():
    assert daily_borrow_return(ETF_COST_PROFILE, weight=-1.0) == pytest.approx(0.0)
    assert daily_borrow_return(ETF_COST_PROFILE, weight=1.0) == pytest.approx(0.0)


def test_config_from_stack_universe_b_defaults():
    cfg = config_from_stack({"UNIVERSE_STAR": "B", "COST_PROFILE_STAR": None})
    assert cfg.cost_profile == ETF_COST_PROFILE
    assert cfg.include_swap is False
    assert cfg.pairs == tuple(ETF_MULTI_TICKERS)


def test_config_from_stack_universe_a_and_event():
    cfg_a = config_from_stack({"UNIVERSE_STAR": "A"})
    assert cfg_a.cost_profile == FX_COST_PROFILE
    assert cfg_a.include_swap is True
    assert "EURUSD" in cfg_a.pairs
    cfg_event = config_from_stack({"COST_PROFILE_STAR": None})
    assert cfg_event.cost_profile == FX_COST_PROFILE
    assert cfg_event.include_swap is True


def test_config_from_stack_cost_profile_star_overrides():
    cfg = config_from_stack(
        {
            "UNIVERSE_STAR": "B",
            "COST_PROFILE_STAR": FX_COST_PROFILE,
            "INCLUDE_SWAP_STAR": True,
        }
    )
    assert cfg.cost_profile == FX_COST_PROFILE
    assert cfg.include_swap is True


def test_simulate_pair_does_not_skip_spy_on_etf_profile():
    dates = pd.bdate_range("2020-01-02", periods=40)
    close = np.linspace(100.0, 110.0, len(dates))
    panel = pd.DataFrame(
        {
            "date": dates,
            "pair": "SPY",
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
        }
    )
    cfg = S3SimConfig(
        pairs=("SPY",),
        cost_profile=ETF_COST_PROFILE,
        include_swap=False,
        tsmom_blend="single_1m",
        tsmom_lookbacks=(21,),
        carry_mode="off",
    )
    res = simulate_pair(panel, cfg)
    assert res.pair == "SPY"
    assert not res.returns_net.empty

    skipped = simulate_pair(panel, S3SimConfig(pairs=("SPY",)))
    assert skipped.pair == ""
    assert skipped.returns_net.empty
