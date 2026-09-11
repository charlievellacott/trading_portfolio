"""Offline unit tests for S3 FX price-panel feature helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.processing.s3_fx_price_panel import add_donchian, add_realised_vol, add_tsmom


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
