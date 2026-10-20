"""Offline unit tests for G10 rate differentials and unit normalization."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ingestion.rates_fetcher import (
    ensure_policy_rates_frame,
    policy_rates_as_annual_decimal,
    rate_differential,
)


def _toy_rates_df_decimal() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "EUR": [0.0, 0.0, 0.0025, 0.0025, 0.0050],
            "USD": [0.0150, 0.0150, 0.0175, 0.0175, 0.0200],
            "JPY": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )


def _toy_rates_df_percent_with_date_col() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "EUR": [0.0, 0.0, 0.25, 0.25, 0.50],
            "USD": [1.50, 1.50, 1.75, 1.75, 2.00],
            "JPY": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_rate_differential_eurusd_synthetic():
    rates = _toy_rates_df_decimal()
    diff = rate_differential("EURUSD", rates)
    expected = rates["EUR"] - rates["USD"]
    pd.testing.assert_series_equal(diff, expected, check_names=False)
    assert diff.name == "EURUSD_rate_diff"
    assert diff.iloc[0] == pytest.approx(-0.0150)
    assert diff.iloc[-1] == pytest.approx(-0.0150)


def test_rate_differential_accepts_slash_pair():
    rates = _toy_rates_df_decimal()
    diff = rate_differential("EUR/USD", rates)
    assert diff.iloc[0] == pytest.approx(-0.0150)


def test_rate_differential_missing_ccy_raises():
    rates = _toy_rates_df_decimal()
    with pytest.raises(KeyError):
        rate_differential("GBPUSD", rates)


def test_policy_rates_percent_to_decimal():
    s = pd.Series([5.17, 5.30, 0.4], name="USD")
    out = policy_rates_as_annual_decimal(s)
    assert out.iloc[0] == pytest.approx(0.0517)
    assert out.iloc[2] == pytest.approx(0.004)


def test_policy_rates_decimal_idempotent():
    s = pd.Series([0.0517, 0.0530, 0.004], name="USD")
    out = policy_rates_as_annual_decimal(s)
    pd.testing.assert_series_equal(out, s)


def test_ensure_policy_rates_frame_date_col_and_percent():
    raw = _toy_rates_df_percent_with_date_col()
    out = ensure_policy_rates_frame(raw)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert "date" not in out.columns
    assert out["USD"].iloc[0] == pytest.approx(0.0150)
    assert out["EUR"].iloc[-1] == pytest.approx(0.0050)
    # Second pass must not divide again.
    again = ensure_policy_rates_frame(out)
    pd.testing.assert_frame_equal(again, out)


def test_rate_differential_accepts_percent_parquet_layout():
    raw = _toy_rates_df_percent_with_date_col()
    diff = rate_differential("EURUSD", raw)
    assert diff.iloc[0] == pytest.approx(-0.0150)
    assert diff.iloc[-1] == pytest.approx(-0.0150)
