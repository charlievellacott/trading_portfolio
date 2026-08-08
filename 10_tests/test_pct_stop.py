"""Tests for risk.pct_stop."""

from __future__ import annotations

import pytest

from risk.pct_stop import pct_stop_price


def test_pct_stop_price_long_20() -> None:
    assert pct_stop_price(200.0, is_long=True, pct=20) == pytest.approx(160.0)


def test_pct_stop_price_short_20() -> None:
    assert pct_stop_price(50.0, is_long=False, pct=20) == pytest.approx(60.0)


def test_pct_stop_price_rejects_nonpositive_entry() -> None:
    with pytest.raises(ValueError, match="entry_px"):
        pct_stop_price(0.0, is_long=True, pct=20)


def test_pct_stop_price_rejects_nonpositive_pct() -> None:
    with pytest.raises(ValueError, match="pct"):
        pct_stop_price(100.0, is_long=True, pct=0)
