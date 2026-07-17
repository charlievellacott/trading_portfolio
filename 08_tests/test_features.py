"""Unit tests for H-001 OBV-confirmed momentum and H-002 GK vol ratio features."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.processing.feature_implementation.gk_vol_ratio import (
    add_gk_vol,
    add_gk_vol_mean,
    add_realised_vol,
    apply_ratio_mode,
    garman_klass_variance,
    garman_klass_vol,
    ratio_from_vols,
    realised_vol,
)
from data.processing.feature_implementation.obv_momentum import (
    add_obv,
    add_obv_trend,
    add_raw_momentum,
    combine_momentum_obv,
    on_balance_volume,
    raw_momentum,
    signs_agree,
)
from data.processing.feature_store import add_gk_vol_ratio, add_obv_confirmed_momentum


def _make_panel(
    n_days: int = 40,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Synthetic long OHLCV panel with deterministic prices/volumes."""
    if tickers is None:
        tickers = ["AAA", "BBB"]
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    frames: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        # Distinct trends: AAA rises, BBB falls then rises.
        base = 100.0 + i * 10.0
        close = base + np.linspace(0, 20, n_days) * (1 if i == 0 else -0.5)
        # Flat day in the middle for AAA
        if ticker == "AAA":
            close = close.copy()
            close[10] = close[9]
        volume = np.full(n_days, 1000.0 * (i + 1), dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def test_raw_momentum_series_formula() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    mom = raw_momentum(close, lookback=3, skip=1)
    # At index 3: close[2]/close[0] - 1 = 12/10 - 1 = 0.2
    assert mom.iloc[3] == pytest.approx(0.2)
    assert pd.isna(mom.iloc[2])


def test_obv_flat_day_unchanged() -> None:
    close = pd.Series([10.0, 11.0, 11.0, 12.0])
    volume = pd.Series([100.0, 200.0, 300.0, 400.0])
    obv = on_balance_volume(close, volume)
    # Day 0: 0; day1: +200; day2 flat: +0; day3: +400
    assert obv.iloc[0] == pytest.approx(0.0)
    assert obv.iloc[1] == pytest.approx(200.0)
    assert obv.iloc[2] == pytest.approx(200.0)
    assert obv.iloc[3] == pytest.approx(600.0)


def test_signs_agree_zero_disagrees() -> None:
    a = pd.Series([1.0, -1.0, 0.0, 1.0])
    b = pd.Series([2.0, -2.0, 3.0, -1.0])
    agree = signs_agree(a, b)
    assert list(agree) == [True, True, False, False]


def test_combine_strict_zero_and_signed() -> None:
    mom = pd.Series([0.1, 0.1, -0.2])
    obv_tr = pd.Series([1.0, -1.0, -2.0])
    strict = combine_momentum_obv(mom, obv_tr, mode="strict_zero")
    signed = combine_momentum_obv(mom, obv_tr, mode="signed")
    assert strict.iloc[0] == pytest.approx(0.1)
    assert strict.iloc[1] == pytest.approx(0.0)
    assert strict.iloc[2] == pytest.approx(-0.2)
    assert signed.iloc[0] == pytest.approx(0.1)
    assert signed.iloc[1] == pytest.approx(-0.1)
    assert signed.iloc[2] == pytest.approx(-0.2)


def test_modular_panel_helpers_add_columns() -> None:
    panel = _make_panel(n_days=30)
    out = add_raw_momentum(panel, lookback=5, skip=1)
    out = add_obv(out)
    out = add_obv_trend(out, obv_window=3)
    assert "raw_momentum" in out.columns
    assert "obv" in out.columns
    assert "obv_trend" in out.columns
    assert out["raw_momentum"].notna().any()
    assert out["obv_trend"].notna().any()


def test_store_auto_column_names_and_modes() -> None:
    panel = _make_panel(n_days=30)
    signed = add_obv_confirmed_momentum(
        panel, lookback=5, skip=1, obv_window=3, mode="signed", normalize=False
    )
    strict = add_obv_confirmed_momentum(
        signed, lookback=5, skip=1, obv_window=3, mode="strict_zero", normalize=False
    )
    assert "obv_mom_signed" in strict.columns
    assert "obv_mom_strict_zero" in strict.columns

    # Where raw mom and obv trend disagree, strict is 0 and signed flips.
    tmp = add_raw_momentum(panel, lookback=5, skip=1)
    tmp = add_obv_trend(tmp, obv_window=3)
    merged = strict.merge(
        tmp[["date", "ticker", "raw_momentum", "obv_trend"]],
        on=["date", "ticker"],
        how="left",
    )
    disagree = (
        merged["raw_momentum"].notna()
        & merged["obv_trend"].notna()
        & ~signs_agree(merged["raw_momentum"], merged["obv_trend"])
    )
    if disagree.any():
        assert (merged.loc[disagree, "obv_mom_strict_zero"] == 0.0).all()
        np.testing.assert_allclose(
            merged.loc[disagree, "obv_mom_signed"].to_numpy(),
            (-merged.loc[disagree, "raw_momentum"]).to_numpy(),
        )


def test_normalize_rank_in_unit_interval() -> None:
    panel = _make_panel(n_days=30, tickers=["AAA", "BBB", "CCC"])
    out = add_obv_confirmed_momentum(
        panel, lookback=5, skip=1, obv_window=3, mode="signed", normalize=True
    )
    vals = out["obv_mom_signed"].dropna()
    assert vals.between(0.0, 1.0).all()


def test_no_lookahead_prefix_stability() -> None:
    panel = _make_panel(n_days=35)
    full = add_obv_confirmed_momentum(
        panel, lookback=5, skip=1, obv_window=3, mode="signed", normalize=False
    )
    cutoff = panel["date"].sort_values().unique()[-5]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_obv_confirmed_momentum(
        truncated, lookback=5, skip=1, obv_window=3, mode="signed", normalize=False
    )
    merged = partial.merge(
        full[["date", "ticker", "obv_mom_signed"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    left = merged["obv_mom_signed_partial"]
    right = merged["obv_mom_signed_full"]
    both = left.notna() & right.notna()
    np.testing.assert_allclose(left[both].to_numpy(), right[both].to_numpy(), rtol=1e-10)


def test_invalid_mode_and_missing_columns() -> None:
    panel = _make_panel(n_days=20)
    with pytest.raises(ValueError, match="mode"):
        add_obv_confirmed_momentum(panel, mode="nope")
    with pytest.raises(ValueError, match="missing columns"):
        add_obv_confirmed_momentum(panel.drop(columns=["volume"]))


# ---------------------------------------------------------------------------
# H-002 GK vol ratio
# ---------------------------------------------------------------------------


def test_garman_klass_one_bar_formula() -> None:
    o = pd.Series([100.0])
    h = pd.Series([110.0])
    lo = pd.Series([90.0])
    c = pd.Series([105.0])
    expected_var = 0.5 * np.log(110 / 90) ** 2 - (2 * np.log(2) - 1) * np.log(105 / 100) ** 2
    var = garman_klass_variance(o, h, lo, c)
    vol = garman_klass_vol(o, h, lo, c)
    assert var.iloc[0] == pytest.approx(expected_var)
    assert vol.iloc[0] == pytest.approx(np.sqrt(max(expected_var, 0.0)))


def test_garman_klass_bad_bar_nan() -> None:
    o = pd.Series([100.0, 100.0])
    h = pd.Series([110.0, 90.0])  # second bar H < L
    lo = pd.Series([90.0, 110.0])
    c = pd.Series([105.0, 100.0])
    var = garman_klass_variance(o, h, lo, c)
    assert var.iloc[0] == pytest.approx(
        0.5 * np.log(110 / 90) ** 2 - (2 * np.log(2) - 1) * np.log(105 / 100) ** 2
    )
    assert pd.isna(var.iloc[1])


def test_realised_vol_and_ratio_series() -> None:
    close = pd.Series([100.0, 101.0, 100.0, 101.0, 100.0, 101.0])
    rv = realised_vol(close, window=3)
    assert pd.isna(rv.iloc[2])  # only 2 returns so far; need 3
    assert rv.iloc[3] == pytest.approx(
        np.std(np.log(close.iloc[1:4].to_numpy() / close.iloc[0:3].to_numpy()))
    )

    short_gk = pd.Series([0.02, 0.04, np.nan])
    realised = pd.Series([0.01, 0.0, 0.02])
    ratio = ratio_from_vols(short_gk, realised)
    assert ratio.iloc[0] == pytest.approx(2.0)
    assert pd.isna(ratio.iloc[1])  # zero denom → NaN (no floor)
    assert pd.isna(ratio.iloc[2])  # NaN short GK


def test_apply_ratio_modes() -> None:
    raw = pd.Series([2.0, 0.5, -1.0])
    assert list(apply_ratio_mode(raw, "ratio")) == pytest.approx([2.0, 0.5, -1.0])
    log_m = apply_ratio_mode(raw, "log_ratio")
    assert log_m.iloc[0] == pytest.approx(np.log(2.0))
    assert log_m.iloc[1] == pytest.approx(np.log(0.5))
    assert pd.isna(log_m.iloc[2])  # non-positive → NaN
    rev = apply_ratio_mode(raw, "reversal")
    np.testing.assert_allclose(rev.to_numpy(), -raw.to_numpy())


def test_gk_modular_panel_helpers() -> None:
    panel = _make_panel(n_days=30)
    out = add_gk_vol(panel)
    out = add_gk_vol_mean(out, gk_window=3)
    out = add_realised_vol(out, realised_window=5)
    assert "gk_vol" in out.columns
    assert "gk_vol_mean" in out.columns
    assert "realised_vol" in out.columns
    assert out["gk_vol"].notna().any()
    assert out["gk_vol_mean"].notna().any()
    assert out["realised_vol"].notna().any()


def test_gk_store_modes_and_column_names() -> None:
    panel = _make_panel(n_days=35)
    ratio = add_gk_vol_ratio(
        panel, gk_window=3, realised_window=5, mode="ratio", normalize=False
    )
    log_r = add_gk_vol_ratio(
        ratio, gk_window=3, realised_window=5, mode="log_ratio", normalize=False
    )
    rev = add_gk_vol_ratio(
        log_r, gk_window=3, realised_window=5, mode="reversal", normalize=False
    )
    assert "gk_vol_ratio" in rev.columns
    assert "gk_vol_log_ratio" in rev.columns
    assert "gk_vol_reversal" in rev.columns

    both = rev["gk_vol_ratio"].notna() & rev["gk_vol_reversal"].notna()
    np.testing.assert_allclose(
        rev.loc[both, "gk_vol_reversal"].to_numpy(),
        (-rev.loc[both, "gk_vol_ratio"]).to_numpy(),
    )
    pos = rev["gk_vol_ratio"] > 0
    np.testing.assert_allclose(
        rev.loc[pos, "gk_vol_log_ratio"].to_numpy(),
        np.log(rev.loc[pos, "gk_vol_ratio"].to_numpy()),
    )


def test_gk_normalize_rank_in_unit_interval() -> None:
    panel = _make_panel(n_days=35, tickers=["AAA", "BBB", "CCC"])
    out = add_gk_vol_ratio(
        panel, gk_window=3, realised_window=5, mode="ratio", normalize=True
    )
    vals = out["gk_vol_ratio"].dropna()
    assert vals.between(0.0, 1.0).all()


def test_gk_no_lookahead_prefix_stability() -> None:
    panel = _make_panel(n_days=40)
    full = add_gk_vol_ratio(
        panel, gk_window=3, realised_window=5, mode="ratio", normalize=False
    )
    cutoff = panel["date"].sort_values().unique()[-5]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_gk_vol_ratio(
        truncated, gk_window=3, realised_window=5, mode="ratio", normalize=False
    )
    merged = partial.merge(
        full[["date", "ticker", "gk_vol_ratio"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    left = merged["gk_vol_ratio_partial"]
    right = merged["gk_vol_ratio_full"]
    both = left.notna() & right.notna()
    np.testing.assert_allclose(left[both].to_numpy(), right[both].to_numpy(), rtol=1e-10)


def test_gk_invalid_mode_and_missing_columns() -> None:
    panel = _make_panel(n_days=25)
    with pytest.raises(ValueError, match="mode"):
        add_gk_vol_ratio(panel, mode="nope")
    with pytest.raises(ValueError, match="missing columns"):
        add_gk_vol_ratio(panel.drop(columns=["high"]))


# --- H-003 / beta regression primitives ---


def test_beta_rolling_ols_hand_check() -> None:
    from data.processing.feature_implementation.beta import rolling_ols_stats

    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 0.01, 30))
    y = 0.001 + 1.5 * x + rng.normal(0, 0.002, 30)
    out = rolling_ols_stats(y, x, 20)
    xw, yw = x.iloc[-20:].to_numpy(), y.iloc[-20:].to_numpy()
    xd, yd = xw - xw.mean(), yw - yw.mean()
    b = (xd * yd).sum() / (xd ** 2).sum()
    a = yw.mean() - b * xw.mean()
    assert out["beta"].iloc[-1] == pytest.approx(b)
    assert out["alpha"].iloc[-1] == pytest.approx(a)
    assert out["beta"].iloc[:19].isna().all()


def test_beta_column_naming_single_vs_multi_window() -> None:
    from data.processing.feature_implementation.beta import (
        add_rolling_beta,
        market_return_frame,
        regression_column_name,
    )

    assert regression_column_name("beta", 20, multi_window=False) == "beta"
    assert regression_column_name("beta", 20, multi_window=True) == "beta_20"

    panel = _make_panel(n_days=35, tickers=["AAA", "BBB"])
    spy = panel[panel["ticker"] == "AAA"][["date", "close"]].copy()
    spy["ticker"] = "SPY"
    mkt = market_return_frame(spy)

    single = add_rolling_beta(panel, mkt, windows=20)
    assert {"alpha", "beta", "r2"}.issubset(single.columns)
    assert "beta_20" not in single.columns
    assert "idio_vol" not in single.columns

    multi = add_rolling_beta(panel, mkt, windows=[10, 20])
    assert {"alpha_10", "beta_10", "r2_10", "alpha_20", "beta_20", "r2_20"}.issubset(
        multi.columns
    )


def test_idiosyncratic_vol_uses_beta_ols() -> None:
    from data.processing.feature_implementation.beta import market_return_frame, rolling_ols_stats
    from data.processing.feature_implementation.idiosyncratic_vol import add_idiosyncratic_vol

    panel = _make_panel(n_days=35, tickers=["AAA", "BBB"])
    spy = panel[panel["ticker"] == "AAA"][["date", "close"]].copy()
    spy["ticker"] = "SPY"
    mkt = market_return_frame(spy)

    out = add_idiosyncratic_vol(panel, mkt, windows=20)
    assert "idio_vol" in out.columns
    assert "idio_vol_20" not in out.columns

    grp = out[out["ticker"] == "AAA"].sort_values("date")
    from data.processing.feature_implementation.beta import log_return

    stats = rolling_ols_stats(
        log_return(grp["close"]),
        mkt.set_index("date").reindex(grp["date"])["market_log_ret"].reset_index(drop=True),
        20,
        include_idio_vol=True,
    )
    both = grp["idio_vol"].notna() & stats["idio_vol"].notna()
    np.testing.assert_allclose(
        grp.loc[both, "idio_vol"].to_numpy(),
        stats.loc[both, "idio_vol"].to_numpy(),
        rtol=1e-10,
    )

