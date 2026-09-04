"""Alpaca shortability helper used by H-001 (mocked TradingClient, no network)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from execution.brokers.alpaca_broker import AlpacaBroker
from data.processing.s2_universe_pools import SHELVED_UNIVERSES


def _broker_with_client(fake_client):
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.client = fake_client
    return broker


class _FakeClient:
    def __init__(self, assets, missing):
        self._assets = assets
        self._missing = set(missing)

    def get_asset(self, symbol):
        if symbol in self._missing:
            raise ValueError(f"asset not found: {symbol}")
        return self._assets[symbol]


def test_shelved_universes_includes_d():
    assert SHELVED_UNIVERSES == frozenset({"A", "B", "C", "D"})


def test_get_asset_shortability_flags_and_missing():
    assets = {
        "WSO.B": SimpleNamespace(
            symbol="WSO.B", shortable=False, easy_to_borrow=False
        ),
        "WSO": SimpleNamespace(symbol="WSO", shortable=True, easy_to_borrow=True),
        "HEI.A": SimpleNamespace(
            symbol="HEI.A", shortable=True, easy_to_borrow=False
        ),
    }
    broker = _broker_with_client(_FakeClient(assets, missing=["MISSING"]))
    rows = broker.get_asset_shortability(["WSO.B", "WSO", "HEI.A", "MISSING"])
    by_ticker = {row["ticker"]: row for row in rows}

    wso_b = by_ticker["WSO.B"]
    assert wso_b["shortable"] is False
    assert wso_b["easy_to_borrow"] is False
    assert wso_b["found"] is True
    assert wso_b["error"] == ""
    assert wso_b["alpaca_symbol"] == "WSO.B"

    wso = by_ticker["WSO"]
    assert wso["shortable"] is True
    assert wso["easy_to_borrow"] is True
    assert wso["found"] is True

    hei_a = by_ticker["HEI.A"]
    assert hei_a["shortable"] is True
    assert hei_a["easy_to_borrow"] is False
    assert hei_a["found"] is True

    missing = by_ticker["MISSING"]
    assert missing["found"] is False
    assert missing["shortable"] is False
    assert missing["easy_to_borrow"] is False
    assert missing["alpaca_symbol"] == ""
    assert "not found" in missing["error"]
