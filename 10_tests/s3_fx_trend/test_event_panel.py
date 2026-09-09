"""Offline unit tests for S3 FX event-panel helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ingestion.fx_fetcher import G10_V1_PAIRS
from data.processing.s3_fx_event_panel import (
    build_s3_event_panel,
    map_event_to_next_bar_fill,
    pair_sign_for_currency,
)


def _toy_calendar(n: int = 30, event_id: str = "us_nfp") -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=n, freq="MS") + pd.Timedelta(hours=13, minutes=30)
    # Deterministic surprises so rolling sigma is well-defined.
    actual = np.linspace(-50.0, 50.0, n)
    forecast = np.zeros(n)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "date": pd.to_datetime(dates).normalize(),
            "event_id": event_id,
            "currency": "USD",
            "impact": 3,
            "actual": actual,
            "forecast": forecast,
            "unit": "k",
        }
    )


def test_shift_before_rolling_std_no_leak():
    cal = _toy_calendar(25)
    panel = build_s3_event_panel(cal, N=5, cache=False)
    # Row 0: no prior surprises → sigma NaN → z NaN
    assert pd.isna(panel["sigma_N"].iloc[0])
    assert pd.isna(panel["z"].iloc[0])
    # Manually recompute: surprise.shift(1).rolling(5).std(ddof=1)
    surprise = panel["surprise"]
    expected_sigma = surprise.shift(1).rolling(5, min_periods=2).std(ddof=1)
    pd.testing.assert_series_equal(
        panel["sigma_N"].reset_index(drop=True),
        expected_sigma.reset_index(drop=True),
        check_names=False,
    )
    # Perturb own-row surprise must not change that row's sigma (uses prior only).
    cal2 = cal.copy()
    cal2.loc[cal2.index[10], "actual"] = 9999.0
    panel2 = build_s3_event_panel(cal2, N=5, cache=False)
    assert panel["sigma_N"].iloc[10] == pytest.approx(panel2["sigma_N"].iloc[10], nan_ok=True)


def test_pair_sign_for_currency_all_seven_pairs():
    assert len(G10_V1_PAIRS) == 7
    # USD is quote on most v1 pairs, base on USDJPY / USDCHF / USDCAD.
    usd_signs = {p: pair_sign_for_currency(p, "USD") for p in G10_V1_PAIRS}
    assert usd_signs["EURUSD"] == -1
    assert usd_signs["GBPUSD"] == -1
    assert usd_signs["AUDUSD"] == -1
    assert usd_signs["NZDUSD"] == -1
    assert usd_signs["USDJPY"] == 1
    assert usd_signs["USDCHF"] == 1
    assert usd_signs["USDCAD"] == 1
    # Base-currency examples
    assert pair_sign_for_currency("EURUSD", "EUR") == 1
    assert pair_sign_for_currency("USDJPY", "JPY") == -1
    # Unrelated ccy
    assert pair_sign_for_currency("EURUSD", "JPY") == 0
    # Every v1 pair has a non-zero USD sign
    assert all(s != 0 for s in usd_signs.values())


def test_map_event_to_next_bar_fill():
    bars = pd.date_range("2020-01-01", periods=5, freq="B")
    # Event during Monday session → next bar is Tuesday
    fill = map_event_to_next_bar_fill("2020-01-01 13:30:00", bars, bar="1d")
    assert fill == pd.Timestamp("2020-01-02")
    # Event exactly on a bar timestamp → strictly after → next bar
    fill2 = map_event_to_next_bar_fill(bars[1], bars, bar="1d")
    assert fill2 == bars[2]
    # After last bar → NaT
    fill3 = map_event_to_next_bar_fill(bars[-1] + pd.Timedelta(days=1), bars)
    assert pd.isna(fill3)
