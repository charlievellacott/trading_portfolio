"""US_ALPACA_D_REALISTIC: tiered slippage and daily borrow on short legs."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.s2_coint.costs import (
    COSTS,
    daily_borrow_return,
    is_alt_share_class_ticker,
    leg_cost_bps,
)


def test_alt_share_class_detection():
    assert is_alt_share_class_ticker("WSO.B")
    assert is_alt_share_class_ticker("HEI.A")
    assert is_alt_share_class_ticker("NWSA")
    assert not is_alt_share_class_ticker("WSO")
    assert not is_alt_share_class_ticker("HEI")


def test_tiered_slippage_common_vs_alt():
    profile = "US_ALPACA_D_REALISTIC"
    common = leg_cost_bps(profile, "WSO", 100.0)
    alt = leg_cost_bps(profile, "WSO.B", 100.0)
    assert alt > common
    assert common == pytest.approx(0.1 + 3.2, rel=1e-6)
    assert alt == pytest.approx(0.1 + 8.0, rel=1e-6)


def test_daily_borrow_on_short_leg_only():
    profile = "US_ALPACA_D_REALISTIC"
    assert daily_borrow_return(profile, y_weight=0.5, x_weight=-0.5) < 0.0
    assert daily_borrow_return(profile, y_weight=0.5, x_weight=0.5) == 0.0
    cfg = COSTS[profile]
    expected = -(100.0 / 252.0 / 10_000.0) * 0.5
    assert daily_borrow_return(profile, y_weight=0.0, x_weight=-0.5) == pytest.approx(
        expected
    )


def test_d_realistic_profile_registered():
    assert "US_ALPACA_D_REALISTIC" in COSTS
    assert COSTS["US_ALPACA_D_REALISTIC"]["borrow_bps_annual"] == 100.0
