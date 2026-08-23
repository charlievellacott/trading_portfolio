"""Tests for S2 universe candidacy and pair-panel builder."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.processing.s2_coint_store import build_pair_panel, screen_pair_cointegration
from data.processing.s2_universe import (
    iter_same_venue_pairs,
    load_s2_pools,
    ticker_venue_key,
)
from data.repo_paths import repo_root


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=n, freq="B")


def _make_coint_pair(
    n: int = 400,
    *,
    alpha: float = 0.5,
    beta: float = 1.2,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    eps = rng.normal(0.0, 0.01, size=n)
    x_log = np.cumsum(eps)
    phi = 0.9
    ou = np.zeros(n)
    noise = rng.normal(0.0, 0.02, size=n)
    for t in range(1, n):
        ou[t] = phi * ou[t - 1] + noise[t]
    y_log = alpha + beta * x_log + ou
    x = pd.Series(np.exp(x_log), index=idx, name="x")
    y = pd.Series(np.exp(y_log), index=idx, name="y")
    return y, x


def _ohlc_from_close(close: pd.Series, *, open_scale: float = 1.0) -> pd.DataFrame:
    """Build an OHLC frame; ``open_scale != 1`` makes open differ from close."""
    c = close.astype(float)
    o = c * float(open_scale)
    return pd.DataFrame(
        {"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c},
        index=c.index,
    )


def test_ticker_venue_key_rules():
    assert ticker_venue_key("0700.HK") == "HK"
    assert ticker_venue_key("8306.T") == "T"
    assert ticker_venue_key("AUDUSD=X") == "FX"
    assert ticker_venue_key("BTC-USD") == "CRYPTO"
    # Plain alphabetic US symbols share one venue so twins are pairable.
    assert ticker_venue_key("GOOGL") == ticker_venue_key("GOOG") == "US"
    # US share-class lines: the dotted segment is a class letter, not an exchange.
    assert ticker_venue_key("BF.A") == ticker_venue_key("BF.B") == "US"
    assert ticker_venue_key("HEI.A") == "US"
    assert ticker_venue_key("CWEN.A") == "US"
    # European exchange suffixes stay distinct from each other.
    assert ticker_venue_key("SAN.MC") == "MC"
    assert ticker_venue_key("ISP.MI") == "MI"
    assert ticker_venue_key("ASML.AS") == "AS"
    assert ticker_venue_key("BMW.DE") == "DE"
    assert ticker_venue_key("MC.PA") == "PA"
    with pytest.raises(ValueError):
        ticker_venue_key("  ")


def test_iter_same_venue_pairs_blocks_cross_venue():
    tickers = ["0700.HK", "9988.HK", "8306.T", "8316.T", "AUDUSD=X", "EURUSD=X"]
    pairs = iter_same_venue_pairs(tickers)
    assert ("0700.HK", "9988.HK") in pairs
    assert ("8306.T", "8316.T") in pairs
    assert ("AUDUSD=X", "EURUSD=X") in pairs
    # No HK↔T
    assert ("0700.HK", "8306.T") not in pairs
    assert ("9988.HK", "8316.T") not in pairs
    # Legs sorted
    assert all(a < b for a, b in pairs)


def test_load_s2_pools_is_dynamic_across_universes():
    """No fixed universe count: the retired CSV loader required exactly three lines."""
    every = load_s2_pools()
    assert set("ABCDEF") <= set(every)
    for label in "ABCDEF":
        pools = load_s2_pools(label)
        assert pools
    with pytest.raises(KeyError):
        load_s2_pools("ZZ")


def test_build_pair_panel_schema_and_metric_flags():
    y, x = _make_coint_pair(n=320, seed=7)
    ohlc = {"AAA": _ohlc_from_close(y), "BBB": _ohlc_from_close(x)}
    pairs = [("AAA", "BBB")]

    base = build_pair_panel(
        ohlc,
        pairs,
        ols_window=40,
        z_window=20,
        hl_window=60,
    )
    expected = [
        "date",
        "pair_id",
        "ticker_y",
        "ticker_x",
        "open_y",
        "high_y",
        "low_y",
        "close_y",
        "open_x",
        "high_x",
        "low_x",
        "close_x",
        "alpha",
        "beta",
        "spread",
        "z",
        "half_life",
    ]
    assert list(base.columns) == expected
    assert "price_y" not in base.columns
    assert "price_x" not in base.columns
    assert base["pair_id"].iloc[0] == "AAA|BBB"
    assert base["z"].notna().sum() > 0

    both = build_pair_panel(
        ohlc,
        pairs,
        ols_window=40,
        z_window=20,
        hl_window=60,
        include_adf_pvalue=True,
        include_variance_jump=True,
    )
    assert "adf_pvalue" in both.columns
    assert "variance_jump" in both.columns

    adf_only = build_pair_panel(
        ohlc,
        pairs,
        ols_window=40,
        z_window=20,
        hl_window=60,
        include_adf_pvalue=True,
        include_variance_jump=False,
    )
    assert "adf_pvalue" in adf_only.columns
    assert "variance_jump" not in adf_only.columns


def test_build_pair_panel_hedge_uses_closes_only():
    y, x = _make_coint_pair(n=320, seed=9)
    # Opens differ from closes; hedge must ignore open/high/low.
    ohlc = {
        "AAA": _ohlc_from_close(y, open_scale=0.97),
        "BBB": _ohlc_from_close(x, open_scale=1.03),
    }
    pairs = [("AAA", "BBB")]
    panel = build_pair_panel(ohlc, pairs, ols_window=40, z_window=20, hl_window=60)

    from data.processing.s2_coint_store import (
        compute_half_life,
        compute_spread_zscore,
        compute_static_hedge_spread,
    )

    hedge = compute_static_hedge_spread(y, x, window=40)
    z = compute_spread_zscore(hedge["spread"], window=20)
    hl = compute_half_life(hedge["spread"], window=60)
    assert np.allclose(panel["spread"].to_numpy(), hedge["spread"].to_numpy(), equal_nan=True)
    assert np.allclose(panel["z"].to_numpy(), z.to_numpy(), equal_nan=True)
    assert np.allclose(panel["half_life"].to_numpy(), hl.to_numpy(), equal_nan=True)
    assert np.allclose(panel["close_y"].to_numpy(), y.to_numpy())
    assert np.allclose(panel["close_x"].to_numpy(), x.to_numpy())
    assert not np.allclose(panel["open_y"].to_numpy(), panel["close_y"].to_numpy())


def test_screen_pair_cointegration_is_end_and_pair_id():
    y, x = _make_coint_pair(n=400, seed=11)
    prices = {"AAA": y, "BBB": x}
    is_end = y.index[int(len(y.index) * 0.7)]
    screened = screen_pair_cointegration(
        prices,
        [("AAA", "BBB")],
        is_end=is_end,
        ols_window=40,
    )
    assert len(screened) == 1
    row = screened.iloc[0]
    assert bool(row["eligible"]) is True
    assert row["n_is_bars"] == int((y.index <= is_end).sum())
    assert "|" in row["pair_id"]
    assert {row["ticker_y"], row["ticker_x"]} == {"AAA", "BBB"}
    assert row["pair_id"] == f"{row['ticker_y']}|{row['ticker_x']}"


def test_screen_pair_cointegration_ineligible_short_overlap():
    y, x = _make_coint_pair(n=400, seed=12)
    # Late-listed x: only 30 mutual bars before a mid-sample is_end.
    x_short = x.iloc[200:].copy()
    is_end = y.index[229]  # 30 bars of x_short on or before is_end
    screened = screen_pair_cointegration(
        {"AAA": y, "BBB": x_short},
        [("AAA", "BBB")],
        is_end=is_end,
        ols_window=40,
    )
    assert len(screened) == 1
    row = screened.iloc[0]
    assert bool(row["eligible"]) is False
    assert row["n_is_bars"] < min(40, 252)
    assert np.isnan(row["pvalue"])
    assert np.isnan(row["tstat"])
    assert np.isnan(row["discovery_half_life"])
