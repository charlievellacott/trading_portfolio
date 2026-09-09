"""Offline unit tests for G10 rate differentials."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ingestion.rates_fetcher import rate_differential


def _toy_rates_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "EUR": [0.0, 0.0, 0.25, 0.25, 0.50],
            "USD": [1.50, 1.50, 1.75, 1.75, 2.00],
            "JPY": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )


def test_rate_differential_eurusd_synthetic():
    rates = _toy_rates_df()
    diff = rate_differential("EURUSD", rates)
    expected = rates["EUR"] - rates["USD"]
    pd.testing.assert_series_equal(diff, expected, check_names=False)
    assert diff.name == "EURUSD_rate_diff"
    assert diff.iloc[0] == pytest.approx(-1.50)
    assert diff.iloc[-1] == pytest.approx(-1.50)


def test_rate_differential_accepts_slash_pair():
    rates = _toy_rates_df()
    diff = rate_differential("EUR/USD", rates)
    assert diff.iloc[0] == pytest.approx(-1.50)


def test_rate_differential_missing_ccy_raises():
    rates = _toy_rates_df()
    with pytest.raises(KeyError):
        rate_differential("GBPUSD", rates)
