"""Tests for H-005 SEC size/valuation fetcher (mocked HTTP; no live SEC)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import data.ingestion.alternative_data.size_value_fetcher as sv_mod
from data.ingestion.alternative_data.size_value_fetcher import (
    DEFAULT_SEC_USER_AGENT,
    _align_fundamentals_to_prices,
    _extract_fundamentals,
    _resolve_user_agent,
    fetch_size_value_daily,
)


def _fixture_companyfacts() -> dict:
    """Minimal CompanyFacts JSON for one ticker."""
    return {
        "cik": 320193,
        "entityName": "Test Co",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "label": "shares",
                    "units": {
                        "shares": [
                            {
                                "end": "2022-12-31",
                                "val": 1000.0,
                                "filed": "2023-01-10",
                                "fp": "FY",
                                "form": "10-K",
                            },
                            {
                                "end": "2023-03-31",
                                "val": 1100.0,
                                "filed": "2023-04-15",
                                "fp": "Q1",
                                "form": "10-Q",
                            },
                        ]
                    },
                }
            },
            "us-gaap": {
                "StockholdersEquity": {
                    "label": "equity",
                    "units": {
                        "USD": [
                            {
                                "end": "2022-12-31",
                                "val": 5000.0,
                                "filed": "2023-01-10",
                                "fp": "FY",
                                "form": "10-K",
                            },
                            {
                                "end": "2023-03-31",
                                "val": 5500.0,
                                "filed": "2023-04-15",
                                "fp": "Q1",
                                "form": "10-Q",
                            },
                        ]
                    },
                },
                "EarningsPerShareDiluted": {
                    "label": "eps",
                    "units": {
                        "USD/shares": [
                            {
                                "end": "2022-12-31",
                                "val": 1.0,
                                "filed": "2023-01-10",
                                "fp": "Q4",
                                "form": "10-K",
                            },
                            {
                                "end": "2023-03-31",
                                "val": 2.0,
                                "filed": "2023-04-15",
                                "fp": "Q1",
                                "form": "10-Q",
                            },
                        ]
                    },
                },
            },
        },
    }


def _tiny_price_panel() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2023-01-05", "2023-01-12", "2023-04-10", "2023-04-20"]
    )
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 4,
            "close": [10.0, 11.0, 12.0, 13.0],
        }
    )


def test_resolve_user_agent_default() -> None:
    assert _resolve_user_agent(None) == DEFAULT_SEC_USER_AGENT
    assert DEFAULT_SEC_USER_AGENT == "trading_portfolio charlie.vellacott@gmail.com"


def test_resolve_user_agent_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _resolve_user_agent("")
    with pytest.raises(ValueError, match="non-empty"):
        _resolve_user_agent("   ")


def test_resolve_user_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "custom_app me@example.com")
    assert _resolve_user_agent(None) == "custom_app me@example.com"
    assert _resolve_user_agent("override@x.com") == "override@x.com"


def test_extract_fundamentals_filing_dated() -> None:
    fund = _extract_fundamentals(_fixture_companyfacts())
    assert list(fund.columns) == [
        "date",
        "shares_outstanding",
        "book_equity",
        "eps_ttm",
    ]
    assert len(fund) == 2
    assert fund["date"].iloc[0] == pd.Timestamp("2023-01-10")
    assert fund["shares_outstanding"].iloc[0] == pytest.approx(1000.0)
    assert fund["book_equity"].iloc[0] == pytest.approx(5000.0)
    # First quarterly EPS observation (min_periods=1 rolling)
    assert fund["eps_ttm"].iloc[0] == pytest.approx(1.0)
    assert fund["eps_ttm"].iloc[1] == pytest.approx(3.0)  # 1 + 2


def test_align_reconstructs_known_ratios() -> None:
    prices = _tiny_price_panel()
    fund = _extract_fundamentals(_fixture_companyfacts())
    out = _align_fundamentals_to_prices(prices, fund)

    # 2023-01-05 is before first filing → NaN fundamentals
    row0 = out.loc[out["date"] == pd.Timestamp("2023-01-05")].iloc[0]
    assert np.isnan(row0["market_cap"])

    # 2023-01-12 uses 2023-01-10 filing: shares=1000, book=5000, eps_ttm=1
    row1 = out.loc[out["date"] == pd.Timestamp("2023-01-12")].iloc[0]
    assert row1["market_cap"] == pytest.approx(11.0 * 1000.0)
    assert row1["pb"] == pytest.approx((11.0 * 1000.0) / 5000.0)
    assert row1["pe"] == pytest.approx(11.0 / 1.0)

    # 2023-04-20 uses 2023-04-15 filing: shares=1100, book=5500, eps_ttm=3
    row3 = out.loc[out["date"] == pd.Timestamp("2023-04-20")].iloc[0]
    assert row3["market_cap"] == pytest.approx(13.0 * 1100.0)
    assert row3["pb"] == pytest.approx((13.0 * 1100.0) / 5500.0)
    assert row3["pe"] == pytest.approx(13.0 / 3.0)


def test_align_no_lookahead_on_truncated_prices() -> None:
    prices = _tiny_price_panel()
    fund = _extract_fundamentals(_fixture_companyfacts())
    full = _align_fundamentals_to_prices(prices, fund)
    prefix = _align_fundamentals_to_prices(prices.iloc[:2], fund)
    shared = full["date"].isin(prefix["date"])
    for col in ("market_cap", "pe", "pb", "shares_outstanding"):
        a = full.loc[shared, col].to_numpy(dtype=float)
        b = prefix[col].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, equal_nan=True)


def test_fetch_size_value_daily_mocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: str,
) -> None:
    cache_dir = str(tmp_path)
    ticker_map = {"0": {"cik_str": 320193, "ticker": "AAA", "title": "Test"}}
    facts = _fixture_companyfacts()
    call_urls: list[str] = []

    def mock_sec_get(url: str, *, user_agent: str):
        call_urls.append(url)
        assert user_agent == DEFAULT_SEC_USER_AGENT
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "company_tickers" in url:
            resp.json.return_value = ticker_map
        elif "companyfacts" in url:
            resp.json.return_value = facts
        else:
            raise AssertionError(f"unexpected url: {url}")
        return resp

    monkeypatch.setattr(sv_mod, "_sec_get", mock_sec_get)
    monkeypatch.setattr(sv_mod, "_REQUEST_SLEEP_SEC", 0.0)

    prices = _tiny_price_panel()
    result = fetch_size_value_daily(
        ["AAA"],
        price_panel=prices,
        cache_dir=cache_dir,
    )
    assert set(result.columns) == {
        "date",
        "ticker",
        "shares_outstanding",
        "book_equity",
        "eps_ttm",
        "market_cap",
        "pe",
        "pb",
    }
    assert len(result) == 4
    assert list(result["ticker"].unique()) == ["AAA"]
    assert result["date"].is_monotonic_increasing

    # Cache written
    assert os.path.exists(os.path.join(cache_dir, "sec_company_tickers.json"))
    assert os.path.exists(
        os.path.join(cache_dir, "sec_companyfacts", "0000320193.json")
    )

    n_calls = len(call_urls)
    result2 = fetch_size_value_daily(
        ["AAA"],
        price_panel=prices,
        cache_dir=cache_dir,
    )
    assert len(call_urls) == n_calls  # no new SEC downloads
    assert len(result2) == 4

    row = result.loc[result["date"] == pd.Timestamp("2023-01-12")].iloc[0]
    assert row["market_cap"] == pytest.approx(11000.0)


def test_fetch_empty_user_agent_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: str,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fetch_size_value_daily(
            ["AAA"],
            price_panel=_tiny_price_panel(),
            cache_dir=str(tmp_path),
            user_agent="",
        )


def test_default_user_agent_on_outbound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: str,
) -> None:
    seen: list[str] = []

    def mock_sec_get(url: str, *, user_agent: str):
        seen.append(user_agent)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "company_tickers" in url:
            resp.json.return_value = {
                "0": {"cik_str": 1, "ticker": "AAA", "title": "A"}
            }
        else:
            resp.json.return_value = {"facts": {}}
        return resp

    monkeypatch.setattr(sv_mod, "_sec_get", mock_sec_get)
    monkeypatch.setattr(sv_mod, "_REQUEST_SLEEP_SEC", 0.0)

    fetch_size_value_daily(
        ["AAA"],
        price_panel=_tiny_price_panel(),
        cache_dir=str(tmp_path),
    )
    assert seen
    assert all(ua == DEFAULT_SEC_USER_AGENT for ua in seen)
