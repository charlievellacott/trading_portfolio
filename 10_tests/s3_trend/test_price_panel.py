"""Offline unit tests for S3 price-panel builders and feature helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.processing.s3_price_panel import (
    ETF_MULTI_TICKERS,
    add_donchian,
    add_realised_vol,
    add_tsmom,
    build_s3_price_panel,
    price_panel_path,
)


def _toy_price_panel(n: int = 80, pair: str = "EURUSD", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 1.10 + np.cumsum(rng.normal(0.0, 0.002, size=n))
    high = close + rng.uniform(0.0005, 0.003, size=n)
    low = close - rng.uniform(0.0005, 0.003, size=n)
    open_ = close + rng.normal(0.0, 0.0005, size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "pair": pair,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def _assert_panel_prefix_stable(
    full: pd.DataFrame, pref: pd.DataFrame, cols: list[str]
) -> None:
    shared_dates = sorted(
        set(pd.to_datetime(full["date"])) & set(pd.to_datetime(pref["date"]))
    )
    for col in cols:
        a = full.loc[pd.to_datetime(full["date"]).isin(shared_dates)].sort_values("date")
        b = pref.loc[pd.to_datetime(pref["date"]).isin(shared_dates)].sort_values("date")
        both = a[col].notna().to_numpy() & b[col].notna().to_numpy()
        if both.any():
            np.testing.assert_allclose(
                a[col].to_numpy(dtype=float)[both],
                b[col].to_numpy(dtype=float)[both],
                rtol=1e-10,
                atol=1e-10,
            )


def test_add_tsmom_donchian_realised_vol_synthetic():
    panel = _toy_price_panel(80)
    out = add_tsmom(panel, lookback_days=10)
    assert f"tsmom_10" in out.columns
    assert out["tsmom_10"].iloc[:10].isna().all()
    assert out["tsmom_10"].iloc[10:].notna().any()

    out = add_donchian(out, N=5)
    for c in ("donch_high_5", "donch_low_5", "donch_break_up_5", "donch_break_dn_5"):
        assert c in out.columns
    # Prior-channel: first N rows of channel should be NaN after shift.
    assert out["donch_high_5"].iloc[:5].isna().all()

    out = add_realised_vol(out, window=10)
    assert "sigma_10" in out.columns
    assert out["sigma_10"].iloc[10:].notna().any()


def test_price_features_no_lookahead_truncated_prefix():
    panel = _toy_price_panel(100)
    full = add_realised_vol(add_donchian(add_tsmom(panel, 12), 8), 15)
    cut = pd.Timestamp(panel["date"].iloc[60])
    trunc = panel.loc[pd.to_datetime(panel["date"]) <= cut].copy()
    pref = add_realised_vol(add_donchian(add_tsmom(trunc, 12), 8), 15)
    _assert_panel_prefix_stable(
        full,
        pref,
        ["tsmom_12", "donch_high_8", "donch_low_8", "sigma_15"],
    )


def test_add_tsmom_rejects_bad_lookback():
    with pytest.raises(ValueError):
        add_tsmom(_toy_price_panel(10), lookback_days=0)


def _fake_etf_ohlcv(ticker, start_date, end_date=None, **kwargs):
    dates = pd.date_range("2010-01-04", periods=4, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.2,
            "volume": 1000,
        }
    )


def _fake_fx_ohlcv_batch(pairs, start, end=None, **kwargs):
    dates = pd.date_range("2010-01-04", periods=3, freq="B")
    rows = []
    for pair in pairs:
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "pair": pair,
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.05,
                }
            )
    return pd.DataFrame(rows)


def test_price_panel_path_has_no_universe_infix(tmp_path):
    path = price_panel_path("1d", data_dir=str(tmp_path))
    assert path.endswith("s3_price_panel_1d.parquet")
    assert "_etf_" not in os.path.basename(path)
    assert "_fx_" not in os.path.basename(path)


def test_build_etf_panel_uses_fetch_ohlcv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "data.processing.s3_price_panel.fetch_ohlcv", _fake_etf_ohlcv
    )

    def _must_not_call_fx(*args, **kwargs):
        raise AssertionError("universe B must not call fetch_fx_ohlcv_batch")

    monkeypatch.setattr(
        "data.processing.s3_price_panel.fetch_fx_ohlcv_batch", _must_not_call_fx
    )
    panel = build_s3_price_panel(
        pairs=("SPY", "TLT"),
        start="2010-01-01",
        end="2010-01-10",
        universe="B",
        cache=True,
        data_dir=str(tmp_path),
    )
    assert set(panel["pair"].unique()) == {"SPY", "TLT"}
    assert list(panel.columns) == ["date", "pair", "open", "high", "low", "close"]
    out = price_panel_path("1d", data_dir=str(tmp_path))
    assert os.path.isfile(out)


def test_build_fx_panel_uses_fx_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "data.processing.s3_price_panel.fetch_fx_ohlcv_batch",
        _fake_fx_ohlcv_batch,
    )

    def _must_not_call_etf(*args, **kwargs):
        raise AssertionError("universe A must not call fetch_ohlcv")

    monkeypatch.setattr("data.processing.s3_price_panel.fetch_ohlcv", _must_not_call_etf)
    panel = build_s3_price_panel(
        pairs=("EURUSD",),
        start="2010-01-01",
        end="2010-01-10",
        universe="A",
        cache=True,
        data_dir=str(tmp_path),
    )
    assert set(panel["pair"].unique()) == {"EURUSD"}
    assert len(panel) == 3


def test_etf_cache_miss_when_file_is_fx(tmp_path, monkeypatch):
    fx = _fake_fx_ohlcv_batch(("EURUSD",), "2010-01-01")
    cache_path = price_panel_path("1d", data_dir=str(tmp_path))
    os.makedirs(tmp_path, exist_ok=True)
    fx.to_parquet(cache_path, index=False)
    called = {"n": 0}

    def _count_etf(ticker, start_date, end_date=None, **kwargs):
        called["n"] += 1
        return _fake_etf_ohlcv(ticker, start_date, end_date)

    monkeypatch.setattr("data.processing.s3_price_panel.fetch_ohlcv", _count_etf)
    panel = build_s3_price_panel(
        pairs=("SPY",),
        start="2010-01-01",
        end="2010-01-10",
        universe="B",
        cache=True,
        data_dir=str(tmp_path),
    )
    assert called["n"] == 1
    assert set(panel["pair"].unique()) == {"SPY"}
    cached = pd.read_parquet(cache_path)
    assert set(cached["pair"].unique()) == {"SPY"}


def test_default_universe_b_instruments(tmp_path, monkeypatch):
    seen: list[str] = []

    def _track(ticker, start_date, end_date=None, **kwargs):
        seen.append(ticker)
        return _fake_etf_ohlcv(ticker, start_date, end_date)

    monkeypatch.setattr("data.processing.s3_price_panel.fetch_ohlcv", _track)
    build_s3_price_panel(
        start="2010-01-01",
        end="2010-01-10",
        universe="B",
        cache=False,
        data_dir=str(tmp_path),
    )
    assert tuple(seen) == ETF_MULTI_TICKERS
