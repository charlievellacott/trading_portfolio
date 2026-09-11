"""Offline unit tests for S3 OANDA cost model."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.s3_fx_trend.costs import (
    COSTS,
    DEFAULT_COST_PROFILE,
    daily_swap_return,
    pair_supported,
)


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
