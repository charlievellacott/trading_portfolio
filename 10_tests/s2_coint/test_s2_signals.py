"""S2 live signals: known z → side, no lookahead, dollar-neutral weights."""

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

from data.processing.feature_implementation.cointegration import to_log_price
from data.processing.s2_coint_store import build_pair_panel, compute_static_hedge_spread
from execution.s2_coint.s2_paper_runner import planned_orders
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.live_decision import walk_live_book
from strategies.s2_coint.sizing import rolling_mean_abs_score


def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def _ohlc_from_close(close: pd.Series) -> pd.DataFrame:
    s = close.astype(float)
    return pd.DataFrame(
        {
            "open": s,
            "high": s,
            "low": s,
            "close": s,
        },
        index=pd.DatetimeIndex(s.index),
    )


def _pair_panel(
    *,
    n: int = 40,
    z: np.ndarray | None = None,
    beta: float = 1.0,
    adf: float = 0.01,
    pair_id: str = "AAA|BBB",
    start: str = "2020-01-01",
) -> pd.DataFrame:
    idx = _dates(n, start)
    if z is None:
        z = np.zeros(n, dtype=float)
    px = np.full(n, 100.0)
    y, x = pair_id.split("|")
    return pd.DataFrame(
        {
            "date": idx,
            "pair_id": pair_id,
            "ticker_y": y,
            "ticker_x": x,
            "open_y": px,
            "high_y": px,
            "low_y": px,
            "close_y": px,
            "open_x": px,
            "high_x": px,
            "low_x": px,
            "close_x": px,
            "alpha": 0.0,
            "beta": beta,
            "spread": 0.0,
            "z": z,
            "half_life": 20.0,
            "adf_pvalue": adf,
            "variance_jump": 1.0,
        }
    )


def _star_like_cfg(**overrides) -> S2SimConfig:
    kwargs = dict(
        hedge="ols",
        bar="1d",
        entry_z=1.5,
        exit_z=0.0,
        z_window=5,
        ols_window=20,
        adf_window=20,
        break_mode="off",
        trend_mode="off",
        overlap_mode="never_allow",
        exit_mode="mean_only",
        size_mode="score",
        vol_mode="fixed_k",
        z_window_mode="fixed",
        entry_mode="trad_z",
        score_column="z",
        beta_column="beta",
    )
    kwargs.update(overrides)
    return S2SimConfig(**kwargs)


def test_rolling_mean_abs_score_window():
    s = pd.Series([1.0, -1.0, 1.0, -1.0, 3.0])
    m = rolling_mean_abs_score(s, 5)
    assert pd.isna(m.iloc[3])
    assert m.iloc[4] == pytest.approx(1.4)


def test_engine_uses_panel_mean_abs_column():
    from strategies.s2_coint.engine import simulate_pair

    z = np.zeros(20, dtype=float)
    z[12:] = 2.0
    panel = _pair_panel(n=20, z=z, beta=1.0)
    panel["mean_abs_score"] = 2.0
    cfg = _star_like_cfg()
    res_col = simulate_pair(panel, cfg, mean_abs_score=99.0)
    panel2 = panel.drop(columns=["mean_abs_score"])
    res_fb = simulate_pair(panel2, cfg, mean_abs_score=2.0)
    assert len(res_col.trades) == len(res_fb.trades)
    if not res_col.trades.empty:
        assert res_col.trades["pnl_pct"].iloc[0] == pytest.approx(
            res_fb.trades["pnl_pct"].iloc[0]
        )


def test_mean_abs_score_summary_smoke():
    from backtest.s2_coint.diagnosis import mean_abs_score_summary

    z = np.linspace(-2.0, 2.0, 30)
    panel = _pair_panel(n=30, z=z)
    out = mean_abs_score_summary(panel, [5, 10], is_end=panel["date"].iloc[-1])
    assert not out.empty
    assert set(out["pair_id"]) == {"AAA|BBB"}
    assert "frozen_is" in set(out["window"].astype(str))


def test_deterministic_z_short_side():
    z = np.zeros(20, dtype=float)
    z[12:] = 2.0
    panel = _pair_panel(n=20, z=z, beta=1.0)
    cfg = _star_like_cfg()
    book = walk_live_book(panel, cfg, universe_tickers=["AAA", "BBB"])
    assert book.open_pos["AAA|BBB"]["side"] == -1
    w = book.weights.set_index("ticker")["weight"]
    assert w["AAA"] < 0
    assert w["BBB"] > 0


def test_deterministic_z_long_side():
    z = np.zeros(20, dtype=float)
    z[12:] = -2.0
    panel = _pair_panel(n=20, z=z, beta=1.0)
    book = walk_live_book(panel, _star_like_cfg(), universe_tickers=["AAA", "BBB"])
    assert book.open_pos["AAA|BBB"]["side"] == 1
    w = book.weights.set_index("ticker")["weight"]
    assert w["AAA"] > 0
    assert w["BBB"] < 0


def test_weights_dollar_neutral_within_tolerance():
    z = np.zeros(20, dtype=float)
    z[12:] = -2.0
    panel = _pair_panel(n=20, z=z, beta=1.0)
    book = walk_live_book(panel, _star_like_cfg(), universe_tickers=["AAA", "BBB"])
    wsum = float(book.weights["weight"].sum())
    assert abs(wsum) < 1e-10


def test_flat_when_abs_z_below_entry():
    z = np.full(20, 1.2)
    panel = _pair_panel(n=20, z=z)
    book = walk_live_book(panel, _star_like_cfg(), universe_tickers=["AAA", "BBB"])
    assert book.open_pos == {}
    assert (book.weights["weight"] == 0).all()


def test_walk_asof_ignores_future_z():
    z = np.zeros(25, dtype=float)
    z[12:20] = 0.0
    z[20:] = 3.0
    panel = _pair_panel(n=25, z=z)
    asof = pd.Timestamp(panel["date"].iloc[19])
    book = walk_live_book(
        panel, _star_like_cfg(), asof=asof, universe_tickers=["AAA", "BBB"]
    )
    assert book.open_pos == {}
    assert book.signal_date == asof


def test_last_row_features_ignore_future_prices():
    rng = np.random.default_rng(7)
    idx = _dates(300)
    x_log = np.cumsum(rng.normal(0.0, 0.01, size=len(idx)))
    y_log = 0.2 + 1.1 * x_log + rng.normal(0.0, 0.02, size=len(idx))
    y = pd.Series(np.exp(y_log), index=idx)
    x = pd.Series(np.exp(x_log), index=idx)
    ohlc_y = _ohlc_from_close(y)
    ohlc_x = _ohlc_from_close(x)
    cut = 250
    full = build_pair_panel(
        {"YYY": ohlc_y, "XXX": ohlc_x},
        [("YYY", "XXX")],
        ols_window=60,
        z_window=40,
        hl_window=80,
        include_adf_pvalue=True,
        include_variance_jump=True,
    )
    pref = build_pair_panel(
        {"YYY": ohlc_y.iloc[:cut], "XXX": ohlc_x.iloc[:cut]},
        [("YYY", "XXX")],
        ols_window=60,
        z_window=40,
        hl_window=80,
        include_adf_pvalue=True,
        include_variance_jump=True,
    )
    last = pd.Timestamp(pref["date"].max())
    full_last = full.loc[pd.to_datetime(full["date"]) == last].iloc[0]
    pref_last = pref.loc[pd.to_datetime(pref["date"]) == last].iloc[0]
    for col in ("z", "beta", "spread", "adf_pvalue"):
        a = float(full_last[col])
        b = float(pref_last[col])
        if np.isfinite(a) and np.isfinite(b):
            assert a == pytest.approx(b, rel=1e-10, abs=1e-10)


def test_prefix_z_matches_full_on_shared_dates():
    rng = np.random.default_rng(11)
    idx = _dates(220)
    x = pd.Series(np.exp(np.cumsum(rng.normal(0.0, 0.01, size=len(idx)))), index=idx)
    y = pd.Series(np.exp(np.log(x) * 1.05 + 0.1), index=idx)
    full = compute_static_hedge_spread(to_log_price(y), to_log_price(x), window=40)
    pref = compute_static_hedge_spread(
        to_log_price(y.iloc[:160]), to_log_price(x.iloc[:160]), window=40
    )
    shared = full["beta"].index.intersection(pref["beta"].index)
    both = full["beta"].loc[shared].notna() & pref["beta"].loc[shared].notna()
    np.testing.assert_allclose(
        full["beta"].loc[shared][both].to_numpy(),
        pref["beta"].loc[shared][both].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_s2_paper_runner_does_not_import_s1():
    from execution.s2_coint import s2_paper_runner as runner

    src = inspect.getsource(runner)
    assert "s1_equities" not in src
    assert "S1Strategy" not in src
    assert "liquidate_all_positions" not in src
    assert "execution.s1_equities" not in src


def test_planned_orders_dry_run_deltas():
    weights = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "weight": [0.25, -0.25],
            "close": [50.0, 50.0],
        }
    )
    plans = planned_orders(
        weights,
        equity=10_000.0,
        current={"AAA": 0.0, "BBB": 0.0},
        universe=["AAA", "BBB"],
    )
    by_t = {p["ticker"]: p for p in plans}
    assert by_t["AAA"]["direction"] == "buy"
    assert by_t["BBB"]["direction"] == "sell"
    assert by_t["AAA"]["quantity"] == 50
    assert by_t["BBB"]["quantity"] == 50


def test_health_flatten_closes_book():
    z = np.zeros(20, dtype=float)
    z[5:15] = -2.0
    z[15:] = -2.0
    panel = _pair_panel(n=20, z=z, adf=0.01)
    panel.loc[15:, "adf_pvalue"] = 0.20
    cfg = _star_like_cfg(break_mode="block_05_flat_10")
    book = walk_live_book(panel, cfg, universe_tickers=["AAA", "BBB"])
    assert book.open_pos == {}
    assert (book.weights["weight"] == 0).all()
