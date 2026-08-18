"""H-001 IS baseline: costs, trad-z simulation, per-pair diagnostics."""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.s2_coint.baseline import (
    clip_ohlc_to_is,
    combine_universe_returns,
    simulate_pair_baseline,
)
from strategies.s2_coint.costs import (
    COSTS,
    leg_cost_bps,
    market_profile_for_pair,
    market_profile_for_ticker,
)
from strategies.s2_coint.metrics import (
    compound_to_s1_weeks,
    corr_to_s1,
    cost_bps_per_year,
    diagnose_locked_panel,
    load_s1_period_returns,
    metrics_from_returns,
    summarize_rolling_adf,
    universe_is_metrics,
)


def _hk_panel(*, n: int = 12, z: np.ndarray | None = None) -> pd.DataFrame:
    """Minimal pair panel with HK legs (constant percent costs)."""
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    if z is None:
        z = np.zeros(n, dtype=float)
        z[0] = -2.5
        z[1] = -2.0
        z[2] = 0.0
    px = np.linspace(10.0, 11.0, n)
    return pd.DataFrame(
        {
            "date": idx,
            "pair_id": "1398.HK|0939.HK",
            "ticker_y": "1398.HK",
            "ticker_x": "0939.HK",
            "open_y": px,
            "high_y": px,
            "low_y": px,
            "close_y": px,
            "open_x": px * 0.5,
            "high_x": px * 0.5,
            "low_x": px * 0.5,
            "close_x": px * 0.5,
            "alpha": 0.0,
            "beta": 2.0,
            "spread": 0.0,
            "z": z,
            "half_life": 20.0,
            "adf_pvalue": np.linspace(0.20, 0.01, n),
        }
    )


def test_market_profile_routing():
    assert market_profile_for_ticker("AUDUSD=X") == "A_FX_OANDA"
    assert market_profile_for_ticker("BNB-USD") == "B_CRYPTO_KRAKEN"
    assert market_profile_for_ticker("1398.HK") == "C_HK_IBKR"
    assert market_profile_for_ticker("8306.T") == "C_JP_IBKR"
    assert market_profile_for_pair("1398.HK|0939.HK", "1398.HK", "0939.HK") == "C_HK_IBKR"
    with pytest.raises(ValueError, match="mixed profiles"):
        market_profile_for_pair("1398.HK|8306.T", "1398.HK", "8306.T")


def test_no_stress_cost_api():
    assert "stress" not in inspect.signature(leg_cost_bps).parameters
    assert "stress_cost" not in inspect.signature(simulate_pair_baseline).parameters
    assert "stress_taker_fee_bps" not in COSTS["B_CRYPTO_KRAKEN"]


def test_hk_leg_cost_is_percent_floor():
    bps = leg_cost_bps("C_HK_IBKR", "1398.HK", 50.0)
    assert bps == pytest.approx(29.0)


def test_clip_ohlc_to_is_drops_later_bars():
    idx = pd.date_range("2021-12-30", periods=4, freq="B")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
        index=idx,
    )
    clipped = clip_ohlc_to_is({"0939.HK": frame}, "2021-12-31")
    assert clipped["0939.HK"].index.max() == pd.Timestamp("2021-12-31")
    assert len(clipped["0939.HK"]) == 2


def test_simulate_records_one_round_trip():
    panel = _hk_panel()
    result = simulate_pair_baseline(panel)
    assert result.pair_id == "1398.HK|0939.HK"
    assert result.n_entries == 1
    assert result.n_open_at_end == 0
    assert len(result.trades) == 1
    row = result.trades.iloc[0]
    assert int(row["side"]) == 1
    assert int(row["hold_bars"]) == 2
    assert float(row["entry_cost_bps"]) == pytest.approx(58.0)
    assert float(row["exit_cost_bps"]) == pytest.approx(58.0)
    assert not result.returns.empty


def test_exit_long_spread_on_z_above_zero():
    """Long spread (z <= -2) flattens on the first bar with z > 0, not exact 0."""
    z = np.zeros(12, dtype=float)
    z[0] = -2.5
    z[1] = -0.4
    z[2] = 0.1
    z[3:] = 0.2
    result = simulate_pair_baseline(_hk_panel(z=z))
    assert result.n_entries == 1
    assert result.n_open_at_end == 0
    assert len(result.trades) == 1
    assert int(result.trades.iloc[0]["side"]) == 1
    assert int(result.trades.iloc[0]["hold_bars"]) == 2


def test_exit_short_spread_on_z_below_zero():
    """Short spread (z >= 2) flattens on the first bar with z < 0."""
    z = np.zeros(12, dtype=float)
    z[0] = 2.5
    z[1] = 0.4
    z[2] = -0.1
    z[3:] = -0.2
    result = simulate_pair_baseline(_hk_panel(z=z))
    assert result.n_entries == 1
    assert result.n_open_at_end == 0
    assert len(result.trades) == 1
    assert int(result.trades.iloc[0]["side"]) == -1
    assert int(result.trades.iloc[0]["hold_bars"]) == 2


def test_per_pair_diagnostics_and_adf():
    panel = _hk_panel()
    table = diagnose_locked_panel(panel, adf_pvalue_threshold=0.05)
    assert list(table["pair_id"]) == ["1398.HK|0939.HK"]
    row = table.iloc[0]
    assert int(row["n_round_trips"]) == 1
    assert int(row["n_entries"]) == 1
    assert float(row["median_hold_bars"]) == pytest.approx(2.0)
    assert float(row["cost_bps_year"]) > 0.0
    assert row["n_adf"] == 12
    assert float(row["last_adf_p"]) == pytest.approx(0.01)
    assert 0.0 < float(row["pct_adf_lt_threshold"]) < 1.0
    assert pd.notna(row["ann_sharpe"])
    assert float(row["max_drawdown"]) <= 0.0


def test_summarize_rolling_adf_empty_and_threshold():
    empty = summarize_rolling_adf(pd.DataFrame())
    assert empty["n_adf"] == 0
    g = pd.DataFrame({"adf_pvalue": [0.20, 0.04, 0.01, np.nan]})
    out = summarize_rolling_adf(g, pvalue_threshold=0.05)
    assert out["n_adf"] == 3
    assert out["last_adf_p"] == pytest.approx(0.01)
    assert out["pct_adf_lt_threshold"] == pytest.approx(2.0 / 3.0)


def test_diagnose_two_pairs_separately():
    a = _hk_panel()
    b = _hk_panel()
    b["pair_id"] = "8306.T|8316.T"
    b["ticker_y"] = "8306.T"
    b["ticker_x"] = "8316.T"
    b["z"] = 0.0
    panel = pd.concat([a, b], ignore_index=True)
    table = diagnose_locked_panel(panel)
    assert set(table["pair_id"]) == {"1398.HK|0939.HK", "8306.T|8316.T"}
    jp = table.loc[table["pair_id"] == "8306.T|8316.T"].iloc[0]
    assert int(jp["n_round_trips"]) == 0
    hk = table.loc[table["pair_id"] == "1398.HK|0939.HK"].iloc[0]
    assert int(hk["n_round_trips"]) == 1


def test_empty_panel_schema():
    table = diagnose_locked_panel(pd.DataFrame())
    assert table.empty
    assert "median_hold_bars" in table.columns
    assert "pct_adf_lt_threshold" in table.columns
    book = universe_is_metrics(pd.DataFrame())
    assert book["n_days"] == 0
    assert np.isnan(book["ann_sharpe"])
    assert np.isnan(book["corr_to_s1"])


def test_metrics_from_returns_and_cost_annualization():
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    ret = pd.Series(0.001, index=idx)
    m = metrics_from_returns(ret)
    assert m["n_days"] == 252
    assert m["ann_sharpe"] > 0
    trades = pd.DataFrame(
        {"entry_cost_bps": [50.0], "exit_cost_bps": [50.0]}
    )
    bps_yr = cost_bps_per_year(ret, trades, open_entry_cost_bps=0.0)
    assert bps_yr == pytest.approx(100.0)


def test_combine_universe_returns_equal_weight():
    a = _hk_panel()
    b = _hk_panel()
    b["pair_id"] = "8306.T|8316.T"
    b["ticker_y"] = "8306.T"
    b["ticker_x"] = "8316.T"
    panel = pd.concat([a, b], ignore_index=True)
    combined = combine_universe_returns(panel)
    ra = simulate_pair_baseline(a).returns
    rb = simulate_pair_baseline(b).returns
    aligned = pd.concat([ra, rb], axis=1).fillna(0.0).mean(axis=1)
    pd.testing.assert_series_equal(
        combined.sort_index(),
        aligned.sort_index().rename("ret"),
        check_names=True,
    )


def test_compound_to_s1_weeks_mon_mon():
    idx = pd.to_datetime(
        [
            "2020-01-06",
            "2020-01-07",
            "2020-01-08",
            "2020-01-09",
            "2020-01-10",
            "2020-01-13",
            "2020-01-14",
            "2020-01-20",
            "2020-01-21",
        ]
    )
    s2 = pd.Series(
        [0.01, 0.02, 0.0, 0.0, 0.03, 0.04, 0.05, 0.10, 0.0],
        index=idx,
    )
    s1_index = pd.DatetimeIndex(["2020-01-06", "2020-01-13", "2020-01-20"])
    weekly = compound_to_s1_weeks(s2, s1_index)
    w0 = (1.01 * 1.02 * 1.0 * 1.0 * 1.03) - 1.0
    w1 = (1.04 * 1.05) - 1.0
    w2 = (1.10 * 1.0) - 1.0
    assert weekly.loc[pd.Timestamp("2020-01-06")] == pytest.approx(w0)
    assert weekly.loc[pd.Timestamp("2020-01-13")] == pytest.approx(w1)
    assert weekly.loc[pd.Timestamp("2020-01-20")] == pytest.approx(w2)


def test_corr_to_s1_perfect_and_missing():
    idx = pd.to_datetime(
        [
            "2020-01-06",
            "2020-01-07",
            "2020-01-08",
            "2020-01-09",
            "2020-01-10",
            "2020-01-13",
            "2020-01-14",
            "2020-01-20",
            "2020-01-21",
        ]
    )
    s2 = pd.Series(
        [0.01, 0.02, 0.0, 0.0, 0.03, 0.04, 0.05, 0.10, 0.0],
        index=idx,
    )
    s1_index = pd.DatetimeIndex(["2020-01-06", "2020-01-13", "2020-01-20"])
    weekly = compound_to_s1_weeks(s2, s1_index)
    assert corr_to_s1(s2, weekly) == pytest.approx(1.0)
    assert np.isnan(corr_to_s1(s2, None))
    assert np.isnan(corr_to_s1(s2, pd.Series(dtype=float)))
    assert np.isnan(corr_to_s1(pd.Series(dtype=float), weekly))
    missing = load_s1_period_returns(os.path.join(ROOT, "no_such_s1_period_returns.parquet"))
    assert missing.empty
    assert np.isnan(universe_is_metrics(pd.DataFrame(), s1_weekly=weekly)["corr_to_s1"])
    book = universe_is_metrics(_hk_panel(), s1_weekly=pd.Series(dtype=float))
    assert "corr_to_s1" in book
    assert np.isnan(book["corr_to_s1"])
