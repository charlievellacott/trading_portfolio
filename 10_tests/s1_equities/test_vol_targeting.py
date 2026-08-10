"""Tests for risk.vol_targeting / signal_conviction and runner wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk.position_sizing import monday_gross_leverage, monday_inv_vol_weights
from risk.signal_conviction import (
    ICScaleConfig,
    bayes_ic_series,
    ic_multiplier_from_history,
    ic_scale_star,
    parse_ic_scale_star,
    update_bayes_ic_state,
    initial_bayes_ic_state,
    ic_from_state,
)
from risk.vol_targeting import (
    ESTIMATOR_BAYES,
    ESTIMATOR_ROLLING,
    VolTargetConfig,
    apply_deadband,
    initial_bayes_vol_state,
    leverage_from_history,
    leverage_series,
    parse_vol_target_star,
    update_bayes_vol_state,
    vol_from_bayes_state,
    vol_target_star,
)


def test_vol_target_star_roundtrip() -> None:
    cfg = VolTargetConfig(
        estimator=ESTIMATOR_BAYES,
        forget=0.94,
        target_ann_vol=0.10,
        quantile=0.5,
        deadband=0.05,
    )
    s = vol_target_star(cfg)
    back = parse_vol_target_star(s)
    assert back.enabled
    assert back.estimator == ESTIMATOR_BAYES
    assert abs(back.forget - 0.94) < 1e-12
    assert abs(back.target_ann_vol - 0.10) < 1e-12
    assert abs(back.deadband - 0.05) < 1e-12
    assert parse_vol_target_star("none").enabled is False


def test_ic_scale_star_roundtrip() -> None:
    cfg = ICScaleConfig(k=25.0, forget=0.94, min_mult=0.25, max_mult=1.25)
    s = ic_scale_star(cfg)
    back = parse_ic_scale_star(s)
    assert back.k == 25.0
    assert abs(back.forget - 0.94) < 1e-12
    assert parse_ic_scale_star("none").enabled is False


def test_apply_deadband() -> None:
    cfg = VolTargetConfig(deadband=0.05)
    assert apply_deadband(1.0, 0.97, cfg) == 1.0
    assert apply_deadband(1.0, 0.80, cfg) == 0.80
    assert apply_deadband(None, 0.90, cfg) == 0.90


def test_high_vol_lower_leverage() -> None:
    cfg = VolTargetConfig(
        estimator=ESTIMATOR_ROLLING,
        window=20,
        min_periods=10,
        target_ann_vol=0.10,
        deadband=0.0,
    )
    rng = np.random.default_rng(0)
    calm = pd.Series(rng.normal(0, 0.005, size=40))
    wild = pd.Series(rng.normal(0, 0.05, size=40))
    l_calm = leverage_from_history(calm, cfg)
    l_wild = leverage_from_history(wild, cfg)
    assert l_wild < l_calm


def test_bayes_quantile_monotonic() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.02, size=40))
    vols = []
    for q in (0.25, 0.5, 0.75):
        cfg = VolTargetConfig(
            estimator=ESTIMATOR_BAYES,
            forget=0.94,
            quantile=q,
            min_periods=10,
        )
        state = initial_bayes_vol_state(cfg)
        for x in r:
            state = update_bayes_vol_state(state, float(x), cfg)
        vols.append(vol_from_bayes_state(state, cfg))
    assert vols[0] <= vols[1] <= vols[2]


def test_scalar_matches_series_last() -> None:
    rng = np.random.default_rng(2)
    r = pd.Series(
        rng.normal(0, 0.015, size=50),
        index=pd.date_range("2020-01-06", periods=50, freq="W-MON"),
    )
    cfg = VolTargetConfig(
        estimator=ESTIMATOR_BAYES,
        forget=0.94,
        min_periods=13,
        deadband=0.0,
    )
    frame = leverage_series(r, cfg)
    # leverage at date i uses returns before i; scalar on r.iloc[:i] matches
    for i in (20, 30, 40):
        past = r.iloc[:i]
        scalar = leverage_from_history(past, cfg)
        # series value at i uses returns[:i] which equals past
        assert abs(scalar - float(frame["leverage"].iloc[i])) < 1e-9


def test_no_lookahead_perturbation() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(
        rng.normal(0, 0.02, size=60),
        index=pd.date_range("2019-01-07", periods=60, freq="W-MON"),
    )
    cfg = VolTargetConfig(estimator=ESTIMATOR_BAYES, forget=0.94, min_periods=13)
    base = leverage_series(r, cfg)["leverage"].copy()
    r2 = r.copy()
    r2.iloc[-5:] = r2.iloc[-5:] + 0.5
    pert = leverage_series(r2, cfg)["leverage"]
    # Early leverage must be unchanged
    assert np.allclose(base.iloc[:-5], pert.iloc[:-5], equal_nan=True)


def test_bayes_incremental_matches_batch() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.02, size=30)
    cfg = VolTargetConfig(estimator=ESTIMATOR_BAYES, forget=0.94, min_periods=5)
    state = initial_bayes_vol_state(cfg)
    for x in r:
        state = update_bayes_vol_state(state, float(x), cfg)
    batch = leverage_from_history(r, cfg)
    # After full history, vol from state → leverage (no deadband)
    vol = vol_from_bayes_state(state, cfg)
    from risk.vol_targeting import _leverage_from_vol

    inc = _leverage_from_vol(vol, cfg)
    assert abs(batch - inc) < 1e-10


def test_ic_shrinks_toward_prior_with_few_names() -> None:
    cfg = ICScaleConfig(
        k=25.0,
        forget=0.94,
        prior_ic=0.0,
        prior_var=0.01,
        min_periods=5,
    )
    # One very high IC with tiny cross-section → strong shrink
    state = initial_bayes_ic_state(cfg)
    state = update_bayes_ic_state(state, 0.9, n_names=6, cfg=cfg)
    post = ic_from_state(state)
    assert abs(post) < 0.9
    assert abs(post) < 0.5


def test_ic_multiplier_clips_negative() -> None:
    cfg = ICScaleConfig(k=25.0, min_mult=0.25, max_mult=1.25, min_periods=1)
    # Force enough history with negative IC
    ics = pd.Series([-0.1] * 20)
    ns = pd.Series([64.0] * 20)
    m = ic_multiplier_from_history(ics, ns, cfg)
    assert m >= cfg.min_mult
    assert m <= cfg.max_mult


def test_monday_gross_leverage_helper() -> None:
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.02, size=40)
    vt = VolTargetConfig(estimator=ESTIMATOR_ROLLING, window=20, min_periods=10)
    out = monday_gross_leverage(r, vt, ic_cfg=ICScaleConfig(enabled=False))
    assert "leverage" in out and out["leverage"] > 0
    assert abs(out["m_ic"] - 1.0) < 1e-12


def test_monday_inv_vol_weights_signed_sleeves() -> None:
    rng = np.random.default_rng(7)
    day_cal = pd.bdate_range("2020-01-01", periods=80)
    tickers = [f"T{i}" for i in range(40)]
    opens = pd.DataFrame(
        100
        * np.exp(
            np.cumsum(rng.normal(0, 0.01, size=(len(day_cal), len(tickers))), axis=0)
        ),
        index=day_cal,
        columns=tickers,
    )
    decision = day_cal[60]
    scores = pd.Series(rng.normal(size=len(tickers)), index=tickers)
    w = monday_inv_vol_weights(
        scores, opens, decision_date=decision, n=15, window=42
    )
    assert not w.empty
    assert float(w[w > 0].sum()) == pytest.approx(0.5, abs=1e-9)
    assert float(w[w < 0].sum()) == pytest.approx(-0.5, abs=1e-9)


def test_runner_none_overlay_matches_baseline() -> None:
    """With overlays off, levered returns equal base path (lev=1)."""
    from backtest.s1_equities.runner import run_backtest
    from backtest.s1_equities.signals import TIMING_MON_OPEN_MON_OPEN

    rng = np.random.default_rng(6)
    dates = pd.date_range("2020-01-06", periods=30, freq="W-MON")
    # Need daily calendar covering holds
    day_cal = pd.bdate_range("2020-01-01", periods=200)
    tickers = [f"T{i}" for i in range(10)]
    scores = pd.DataFrame(
        rng.normal(size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    opens = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(day_cal), len(tickers))), axis=0)),
        index=day_cal,
        columns=tickers,
    )
    closes = opens * (1 + rng.normal(0, 0.001, size=opens.shape))
    highs = np.maximum(opens, closes) * 1.01
    lows = np.minimum(opens, closes) * 0.99
    highs = pd.DataFrame(highs, index=day_cal, columns=tickers)
    lows = pd.DataFrame(lows, index=day_cal, columns=tickers)

    a = run_backtest(
        scores,
        opens,
        closes,
        n=3,
        timing_mode=TIMING_MON_OPEN_MON_OPEN,
        highs=highs,
        lows=lows,
    )
    b = run_backtest(
        scores,
        opens,
        closes,
        n=3,
        timing_mode=TIMING_MON_OPEN_MON_OPEN,
        highs=highs,
        lows=lows,
        vol_target=VolTargetConfig(enabled=False),
        ic_scale=ICScaleConfig(enabled=False),
    )
    assert np.allclose(a.period_returns, b.period_returns, equal_nan=True)
    assert np.allclose(a.turnover, b.turnover, equal_nan=True)
    assert (b.leverage.fillna(1.0) == 1.0).all()


def test_runner_vol_target_reduces_gross_in_high_vol() -> None:
    from backtest.s1_equities.runner import run_backtest
    from backtest.s1_equities.signals import TIMING_MON_OPEN_MON_OPEN

    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-01", periods=80, freq="W-MON")
    day_cal = pd.bdate_range("2017-12-01", periods=600)
    tickers = [f"T{i}" for i in range(12)]
    scores = pd.DataFrame(
        rng.normal(size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    # High daily vol → wild weekly strategy returns
    opens = pd.DataFrame(
        100
        * np.exp(
            np.cumsum(rng.normal(0, 0.04, size=(len(day_cal), len(tickers))), axis=0)
        ),
        index=day_cal,
        columns=tickers,
    )
    closes = opens.copy()
    highs = opens * 1.02
    lows = opens * 0.98
    vt = VolTargetConfig(
        estimator=ESTIMATOR_ROLLING,
        window=13,
        min_periods=8,
        target_ann_vol=0.10,
        max_leverage=1.5,
        min_leverage=0.25,
        deadband=0.0,
    )
    res = run_backtest(
        scores,
        opens,
        closes,
        n=3,
        timing_mode=TIMING_MON_OPEN_MON_OPEN,
        highs=highs,
        lows=lows,
        vol_target=vt,
    )
    assert res.leverage.notna().any()
    # After warmup, leverage should often sit below 1 in this noisy world
    late = res.leverage.iloc[20:]
    assert late.mean() < 1.0
