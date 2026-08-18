"""Offline tests for S2 cointegration math and store dispatchers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.processing.feature_implementation.cointegration import (
    is_integrated_order_one,
    kalman_hedge,
    ou_half_life,
    residual_variance_ratio,
    rolling_adf_pvalue,
    rolling_hedge,
    rolling_ou_half_life,
    rolling_zscore,
    test_cointegration,
    to_log_price,
)
from data.processing.feature_implementation.kalman import kalman_linear_regression
from data.processing.s2_coint_store import (
    COINT_METRICS,
    _validate_pair_inputs,
    compute_coint_metrics,
    compute_half_life,
    compute_kalman_hedge_spread,
    compute_spread_zscore,
    compute_static_hedge_spread,
    run_cointegration_test,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=n, freq="B")


def _make_coint_pair(
    n: int = 800,
    *,
    alpha: float = 0.5,
    beta: float = 1.2,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """x = RW; y = alpha + beta*x + OU noise (in log space, returned as raw prices)."""
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    eps = rng.normal(0.0, 0.01, size=n)
    x_log = np.cumsum(eps)
    # OU residual
    phi = 0.9
    ou = np.zeros(n)
    noise = rng.normal(0.0, 0.02, size=n)
    for t in range(1, n):
        ou[t] = phi * ou[t - 1] + noise[t]
    y_log = alpha + beta * x_log + ou
    x = pd.Series(np.exp(x_log), index=idx, name="x")
    y = pd.Series(np.exp(y_log), index=idx, name="y")
    return y, x


def _make_independent_pair(
    n: int = 800,
    *,
    seed: int = 1,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    x = pd.Series(np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n))), index=idx, name="x")
    y = pd.Series(np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n))), index=idx, name="y")
    return y, x


def _ar1_spread(n: int, phi: float, *, seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0.0, 0.05)
    return pd.Series(s, index=idx)


# ---------------------------------------------------------------------------
# Discovery / I(1)
# ---------------------------------------------------------------------------


def test_is_integrated_order_one_random_walk_true_white_noise_false():
    rng = np.random.default_rng(3)
    idx = _dates(500)
    rw = pd.Series(np.cumsum(rng.normal(0.0, 1.0, size=500)), index=idx)
    wn = pd.Series(rng.normal(0.0, 1.0, size=500), index=idx)
    assert is_integrated_order_one(rw)[0] is True
    assert is_integrated_order_one(wn)[0] is False


def test_engle_granger_detects_coint_pair_and_recovers_beta():
    y, x = _make_coint_pair(alpha=0.5, beta=1.2, seed=10)
    y_log, x_log = to_log_price(y), to_log_price(x)
    result = test_cointegration(y_log, x_log)
    assert result.is_cointegrated is True
    assert result.pvalue < 0.05
    assert result.direction in ("y~x", "x~y")
    if result.direction == "y~x":
        assert result.beta == pytest.approx(1.2, rel=0.15, abs=0.2)
    else:
        # inverse hedge ≈ 1/1.2
        assert result.beta == pytest.approx(1.0 / 1.2, rel=0.2, abs=0.25)


def test_engle_granger_rejects_independent_pair():
    y, x = _make_independent_pair(seed=11)
    result = test_cointegration(to_log_price(y), to_log_price(x))
    assert result.is_cointegrated is False


def test_direction_populates_and_matches_ols():
    y, x = _make_coint_pair(seed=12)
    result = test_cointegration(to_log_price(y), to_log_price(x))
    assert result.direction in ("y~x", "x~y")
    assert np.isfinite(result.alpha)
    assert np.isfinite(result.beta)


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


def test_ou_half_life_hand_check_known_phi():
    # AR(1): s_t = phi * s_{t-1} + e  =>  Δs = (phi-1) s_{t-1} + e  => b = phi-1
    # hl = -ln(2)/ln(phi)
    phi = 0.9
    expected = -np.log(2.0) / np.log(phi)
    s = _ar1_spread(2000, phi, seed=20)
    hl = ou_half_life(s)
    assert hl == pytest.approx(expected, rel=0.25)


def test_ou_half_life_random_walk_nan():
    # Strict unit-root path (deterministic trend in levels) => b ~ 0 / non-MR
    s = pd.Series(np.arange(500, dtype=float), index=_dates(500))
    assert np.isnan(ou_half_life(s))


def test_ou_half_life_oscillatory_nan():
    # phi < 0 => 1+b = phi could be <= 0 when |phi| large; use phi = -0.5 => b = -1.5, 1+b=-0.5 < 0? 
    # Actually phi=-0.5 => b=phi-1=-1.5, 1+b=-0.5 <= 0 → NaN. Good.
    # Or phi=-1.2 oscillatory exploding - use alternating
    s = pd.Series(
        [((-1.0) ** t) * 1.0 for t in range(200)],
        index=_dates(200),
    )
    assert np.isnan(ou_half_life(s))


# ---------------------------------------------------------------------------
# No-lookahead prefix stability
# ---------------------------------------------------------------------------


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


def test_no_lookahead_rolling_hedge():
    y, x = _make_coint_pair(n=400, seed=30)
    y_log, x_log = to_log_price(y), to_log_price(x)
    full = rolling_hedge(y_log, x_log, window=60)
    cut = 250
    pref = rolling_hedge(y_log.iloc[:cut], x_log.iloc[:cut], window=60)
    _assert_prefix_stable(full["spread"], pref["spread"])
    _assert_prefix_stable(full["beta"], pref["beta"])


def test_no_lookahead_rolling_zscore():
    y, x = _make_coint_pair(n=400, seed=31)
    spread = rolling_hedge(to_log_price(y), to_log_price(x), window=60)["spread"]
    full = rolling_zscore(spread, window=40)
    cut = 250
    pref = rolling_zscore(spread.iloc[:cut], window=40)
    _assert_prefix_stable(full, pref)


def test_no_lookahead_rolling_adf_pvalue():
    y, x = _make_coint_pair(n=350, seed=32)
    spread = rolling_hedge(to_log_price(y), to_log_price(x), window=50)["spread"]
    full = rolling_adf_pvalue(spread, window=80)
    cut = 280
    pref = rolling_adf_pvalue(spread.iloc[:cut], window=80)
    _assert_prefix_stable(full, pref)


def test_no_lookahead_residual_variance_ratio():
    y, x = _make_coint_pair(n=400, seed=33)
    spread = rolling_hedge(to_log_price(y), to_log_price(x), window=50)["spread"]
    full = residual_variance_ratio(spread, window=20, baseline_window=60)
    cut = 300
    pref = residual_variance_ratio(spread.iloc[:cut], window=20, baseline_window=60)
    _assert_prefix_stable(full, pref)


def test_no_lookahead_kalman_linear_regression():
    y, x = _make_coint_pair(n=300, seed=34)
    y_log, x_log = to_log_price(y), to_log_price(x)
    full = kalman_linear_regression(y_log, x_log, burn_in=20)
    cut = 200
    pref = kalman_linear_regression(y_log.iloc[:cut], x_log.iloc[:cut], burn_in=20)
    _assert_prefix_stable(full["spread"], pref["spread"])
    _assert_prefix_stable(full["beta"], pref["beta"])


# ---------------------------------------------------------------------------
# Kalman
# ---------------------------------------------------------------------------


def test_kalman_recovers_constant_beta_after_burn_in():
    rng = np.random.default_rng(40)
    n = 500
    idx = _dates(n)
    true_beta, true_alpha = 1.5, 0.2
    x_log = pd.Series(np.cumsum(rng.normal(0.0, 0.01, size=n)), index=idx)
    y_log = true_alpha + true_beta * x_log + rng.normal(0.0, 0.01, size=n)
    out = kalman_linear_regression(y_log, x_log, delta=1e-5, obs_var=1e-3, burn_in=50)
    assert out["beta"].iloc[:50].isna().all()
    # After burn-in, mean beta near truth
    assert out["beta"].iloc[100:].mean() == pytest.approx(true_beta, rel=0.1, abs=0.15)


# ---------------------------------------------------------------------------
# Variance jump
# ---------------------------------------------------------------------------


def test_residual_variance_ratio_rises_on_regime_change():
    rng = np.random.default_rng(50)
    n = 500
    idx = _dates(n)
    s = np.concatenate(
        [
            rng.normal(0.0, 0.05, size=300),
            rng.normal(0.0, 0.4, size=200),
        ]
    )
    spread = pd.Series(s, index=idx)
    ratio = residual_variance_ratio(spread, window=40, baseline_window=100)
    early = ratio.iloc[200:280].median()
    late = ratio.iloc[380:450].median()
    assert late > early
    assert late > 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_to_log_price_nonpositive_nan():
    s = pd.Series([1.0, 0.0, -2.0, np.nan, 3.0], index=_dates(5))
    out = to_log_price(s)
    assert np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[2])
    assert np.isnan(out.iloc[3])
    assert np.isfinite(out.iloc[0])
    assert np.isfinite(out.iloc[4])


def test_rolling_zscore_zero_std_nan_and_no_bfill():
    idx = _dates(100)
    # constant after warm-up of NaNs at start
    s = pd.Series([np.nan] * 10 + [1.0] * 90, index=idx)
    z = rolling_zscore(s, window=20)
    # leading NaNs preserved (no bfill)
    assert z.iloc[:10].isna().all()
    # constant stretch → std 0 → NaN
    assert z.iloc[40:].isna().all()


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------


def test_compute_coint_metrics_selector():
    y, x = _make_coint_pair(n=300, seed=60)
    spread = rolling_hedge(to_log_price(y), to_log_price(x), window=40)["spread"]
    both = compute_coint_metrics(
        spread, metrics=None, adf_window=80, var_window=20, var_baseline_window=60
    )
    assert list(both.columns) == list(COINT_METRICS)

    one = compute_coint_metrics(
        spread, metrics=["variance_jump"], var_window=20, var_baseline_window=60
    )
    assert list(one.columns) == ["variance_jump"]

    dedup = compute_coint_metrics(
        spread,
        metrics=["adf_pvalue", "adf_pvalue"],
        adf_window=80,
    )
    assert list(dedup.columns) == ["adf_pvalue"]

    with pytest.raises(ValueError, match="unknown"):
        compute_coint_metrics(spread, metrics=["not_a_metric"])


def test_store_matches_math_on_log_prices():
    y, x = _make_coint_pair(n=250, seed=61)
    y_log, x_log = to_log_price(y), to_log_price(x)

    store_static = compute_static_hedge_spread(y, x, window=40)
    math_static = rolling_hedge(y_log, x_log, window=40)
    pd.testing.assert_frame_equal(store_static, math_static)

    store_kf = compute_kalman_hedge_spread(y, x, burn_in=15)
    math_kf = kalman_hedge(y_log, x_log, burn_in=15)
    pd.testing.assert_frame_equal(store_kf, math_kf)

    spread = math_static["spread"]
    pd.testing.assert_series_equal(
        compute_spread_zscore(spread, window=30),
        rolling_zscore(spread, window=30),
    )
    pd.testing.assert_series_equal(
        compute_half_life(spread, window=80),
        rolling_ou_half_life(spread, window=80),
    )


def test_validate_pair_inputs_errors():
    y, x = _make_coint_pair(n=50, seed=62)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        _validate_pair_inputs(pd.Series(y.to_numpy()), x)

    bad = y.copy()
    bad.index = bad.index[::-1]
    with pytest.raises(ValueError, match="monotonic"):
        _validate_pair_inputs(bad, x)

    dup_idx = y.index.append(pd.DatetimeIndex([y.index[-1]]))
    dup_y = pd.Series(np.arange(len(dup_idx), dtype=float), index=dup_idx)
    dup_x = pd.Series(np.arange(len(dup_idx), dtype=float), index=dup_idx)
    with pytest.raises(ValueError, match="duplicates"):
        _validate_pair_inputs(dup_y, dup_x)

    with pytest.raises(ValueError, match="identical"):
        _validate_pair_inputs(y, x.iloc[:-1])


def test_run_cointegration_test_accepts_raw_prices():
    y, x = _make_coint_pair(n=600, seed=63)
    store_res = run_cointegration_test(y, x)
    math_res = test_cointegration(to_log_price(y), to_log_price(x))
    assert store_res.is_cointegrated == math_res.is_cointegrated
    assert store_res.direction == math_res.direction
    assert store_res.pvalue == pytest.approx(math_res.pvalue, rel=1e-10, abs=1e-12)
