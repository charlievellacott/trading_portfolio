"""Offline unit tests for BIS REER helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ingestion.alternative_data.bis_reer import (
    BIS_REER_SERIES,
    pair_reer_value_signal,
)
from data.ingestion.fx_fetcher import G10_V1_PAIRS


def _toy_reer_df(n_months: int = 48) -> pd.DataFrame:
    """Synthetic monthly REER with rich EUR vs USD in the late sample."""
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    # Flat early, then EUR REER rises (rich base) vs USD.
    eur = np.full(n_months, 100.0)
    usd = np.full(n_months, 100.0)
    eur[-12:] = np.linspace(110.0, 130.0, 12)
    usd[-12:] = np.linspace(100.0, 102.0, 12)
    df = pd.DataFrame({"EUR": eur, "USD": usd}, index=idx)
    df.index.name = "date"
    df["availability_date"] = pd.DatetimeIndex(df.index) + pd.offsets.MonthBegin(2)
    return df


def test_bis_reer_series_map_covers_g10_v1_currencies():
    assert len(G10_V1_PAIRS) == 7
    ccys = set()
    for pair in G10_V1_PAIRS:
        ccys.add(pair[:3])
        ccys.add(pair[3:])
    # 7 v1 pairs span 8 ISO codes (USD + 7 crosses); map covers each.
    assert len(ccys) == 8
    for ccy in ccys:
        assert ccy in BIS_REER_SERIES
    assert len(BIS_REER_SERIES) == 8


def test_availability_date_june_to_august_first():
    june_end = pd.Timestamp("2020-06-30")
    avail = june_end + pd.offsets.MonthBegin(2)
    assert avail == pd.Timestamp("2020-08-01")
    # Same rule as fetch_bis_reer attaches on month-end index.
    df = pd.DataFrame({"USD": [100.0]}, index=pd.DatetimeIndex([june_end]))
    df["availability_date"] = pd.DatetimeIndex(df.index) + pd.offsets.MonthBegin(2)
    assert df["availability_date"].iloc[0] == pd.Timestamp("2020-08-01")


def test_pair_reer_value_signal_rich_base_negative():
    reer = _toy_reer_df(48)
    sig = pair_reer_value_signal("EURUSD", reer, z_window=24)
    assert sig.dropna().shape[0] > 0
    # Late sample: EUR rich vs USD → negative lean for long-EUR.
    late = sig.dropna().iloc[-30:]
    assert float(late.mean()) < 0.0


def _assert_prefix_stable(full: pd.Series, prefix: pd.Series) -> None:
    shared = full.index.intersection(prefix.index)
    a = full.loc[shared]
    b = prefix.loc[shared]
    both = a.notna() & b.notna()
    if both.any():
        np.testing.assert_allclose(
            a.loc[both].to_numpy(dtype=float),
            b.loc[both].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-10,
        )


def test_pair_reer_value_signal_no_lookahead_truncated_prefix():
    reer = _toy_reer_df(60)
    full = pair_reer_value_signal("EURUSD", reer, z_window=24)
    cut = reer.index[40]
    pref = pair_reer_value_signal(
        "EURUSD", reer.loc[reer.index <= cut].copy(), z_window=24
    )
    _assert_prefix_stable(full, pref)
