"""Unit tests for Yahoo GICS helper (no network)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from data.ingestion.equity_gics import GICS_COLUMNS, fetch_yahoo_gics


class _FakeTicker:
    def __init__(self, symbol: str, info: dict):
        self._symbol = symbol
        self.info = info


def test_fetch_yahoo_gics_schema_and_missing_labels(tmp_path, monkeypatch):
    cache = os.path.join(str(tmp_path), "yahoo_gics.parquet")
    infos = {
        "AAA": {"sector": "Technology", "industry": "Software"},
        "BBB": {"sector": None, "industry": ""},
        "CCC": {},
    }

    def fake_ticker(symbol: str):
        return _FakeTicker(symbol, infos.get(symbol.upper(), {}))

    monkeypatch.setattr("data.ingestion.equity_gics.yf.Ticker", fake_ticker)

    out = fetch_yahoo_gics(["aaa", "BBB", "CCC"], cache_path=cache)
    assert list(out.columns) == list(GICS_COLUMNS)
    assert list(out["ticker"]) == ["AAA", "BBB", "CCC"]
    assert out.loc[out["ticker"] == "AAA", "sector"].iloc[0] == "Technology"
    assert out.loc[out["ticker"] == "AAA", "industry"].iloc[0] == "Software"
    assert out.loc[out["ticker"] == "BBB", "sector"].iloc[0] == ""
    assert out.loc[out["ticker"] == "BBB", "industry"].iloc[0] == ""
    assert out.loc[out["ticker"] == "CCC", "sector"].iloc[0] == ""
    assert out.loc[out["ticker"] == "CCC", "industry"].iloc[0] == ""
    assert os.path.isfile(cache)

    # Cache hit: do not call Yahoo again for known tickers.
    calls: list[str] = []

    def counting_ticker(symbol: str):
        calls.append(symbol)
        return _FakeTicker(symbol, {"sector": "ShouldNotUse", "industry": "X"})

    monkeypatch.setattr("data.ingestion.equity_gics.yf.Ticker", counting_ticker)
    again = fetch_yahoo_gics(["AAA", "BBB"], cache_path=cache)
    assert calls == []
    assert again.loc[again["ticker"] == "AAA", "sector"].iloc[0] == "Technology"


def test_fetch_yahoo_gics_empty_request(tmp_path):
    cache = os.path.join(str(tmp_path), "yahoo_gics.parquet")
    out = fetch_yahoo_gics([], cache_path=cache)
    assert out.empty
    assert list(out.columns) == list(GICS_COLUMNS)
