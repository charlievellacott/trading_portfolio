"""Offline unit tests for ``data.ingestion.fx_fetcher``."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ingestion.credentials_env import read_credential
from data.ingestion.fx_fetcher import (
    G10_V1_PAIRS,
    _normalize_pair,
    fetch_fx_ohlcv,
    fetch_fx_ohlcv_yf,
)


def test_g10_v1_pairs_length_seven():
    assert len(G10_V1_PAIRS) == 7
    assert G10_V1_PAIRS[0] == "EURUSD"
    assert "NZDUSD" in G10_V1_PAIRS


def test_normalize_pair_helpers_and_public_api_errors():
    assert _normalize_pair("eur/usd") == "EURUSD"
    assert _normalize_pair("EUR_USD") == "EURUSD"
    assert _normalize_pair("EURUSD=X") == "EURUSD"
    with pytest.raises(ValueError, match="6-letter"):
        fetch_fx_ohlcv("EU", "2020-01-01", "2020-01-05", source="yfinance")
    with pytest.raises(ValueError, match="6-letter"):
        fetch_fx_ohlcv("EURUSDJPY", "2020-01-01", "2020-01-05", source="yfinance")
    with pytest.raises(ValueError, match="6-letter"):
        _normalize_pair("EUR/US")


@pytest.mark.skipif(
    (read_credential("OANDA_API_TOKEN") or "keyhere") == "keyhere",
    reason="OANDA_API_TOKEN missing or placeholder keyhere",
)
def test_oanda_fetch_smoke_when_token_present():
    df = fetch_fx_ohlcv(
        "EURUSD",
        "2024-01-02",
        "2024-01-10",
        interval="1d",
        source="oanda",
    )
    assert not df.empty
    assert {"date", "open", "high", "low", "close", "pair"}.issubset(df.columns)


@pytest.mark.skip(reason="network yfinance smoke — run manually when needed")
def test_yfinance_smoke_skipped_by_default():
    df = fetch_fx_ohlcv_yf("EURUSD", "2024-01-02", "2024-01-10", interval="1d")
    assert not df.empty
    assert df["pair"].iloc[0] == "EURUSD"
