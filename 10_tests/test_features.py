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
from data.processing.feature_store import (
    add_gk_vol_ratio,
    add_idiosyncratic_vol,
    add_obv_confirmed_momentum,
)


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


def test_obv_multi_window_column_names_and_parity() -> None:
    from data.processing.feature_implementation.beta import windowed_column_name

    assert windowed_column_name("obv_mom_signed", 5, 1, 3, multi=False) == "obv_mom_signed"
    assert (
        windowed_column_name("obv_mom_signed", 5, 1, 3, multi=True) == "obv_mom_signed_5_1_3"
    )

    panel = _make_panel(n_days=40)
    multi = add_obv_confirmed_momentum(
        panel,
        lookback=[5, 8],
        skip=1,
        obv_window=3,
        mode="signed",
        normalize=False,
    )
    assert "obv_mom_signed_5_1_3" in multi.columns
    assert "obv_mom_signed_8_1_3" in multi.columns
    assert "obv_mom_signed" not in multi.columns

    single_5 = add_obv_confirmed_momentum(
        panel, lookback=5, skip=1, obv_window=3, mode="signed", normalize=False
    )
    both = multi["obv_mom_signed_5_1_3"].notna() & single_5["obv_mom_signed"].notna()
    np.testing.assert_allclose(
        multi.loc[both, "obv_mom_signed_5_1_3"].to_numpy(),
        single_5.loc[both, "obv_mom_signed"].to_numpy(),
        rtol=1e-10,
    )

    with pytest.raises(ValueError, match="lookback must be greater than skip"):
        add_obv_confirmed_momentum(panel, lookback=[5], skip=[5], obv_window=3)


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


def test_gk_multi_window_column_names_and_parity() -> None:
    panel = _make_panel(n_days=45)
    multi = add_gk_vol_ratio(
        panel,
        gk_window=[3, 4],
        realised_window=5,
        mode="ratio",
        normalize=False,
    )
    assert "gk_vol_ratio_3_5" in multi.columns
    assert "gk_vol_ratio_4_5" in multi.columns
    assert "gk_vol_ratio" not in multi.columns

    single = add_gk_vol_ratio(
        panel, gk_window=3, realised_window=5, mode="ratio", normalize=False
    )
    both = multi["gk_vol_ratio_3_5"].notna() & single["gk_vol_ratio"].notna()
    np.testing.assert_allclose(
        multi.loc[both, "gk_vol_ratio_3_5"].to_numpy(),
        single.loc[both, "gk_vol_ratio"].to_numpy(),
        rtol=1e-10,
    )


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


def _idio_market_frame(panel: pd.DataFrame) -> pd.DataFrame:
    from data.processing.feature_implementation.beta import market_return_frame

    spy = panel[panel["ticker"] == panel["ticker"].iloc[0]][["date", "close"]].copy()
    spy["ticker"] = "SPY"
    return market_return_frame(spy)


def test_idio_store_single_vs_multi_window_parity() -> None:
    panel = _make_panel(n_days=40, tickers=["AAA", "BBB", "CCC"])
    mkt = _idio_market_frame(panel)

    multi = add_idiosyncratic_vol(panel, mkt, windows=[15, 20], normalize=False)
    assert "idio_vol_15" in multi.columns
    assert "idio_vol_20" in multi.columns
    assert "idio_vol" not in multi.columns

    single = add_idiosyncratic_vol(panel, mkt, windows=20, normalize=False)
    assert "idio_vol" in single.columns
    both = multi["idio_vol_20"].notna() & single["idio_vol"].notna()
    np.testing.assert_allclose(
        multi.loc[both, "idio_vol_20"].to_numpy(),
        single.loc[both, "idio_vol"].to_numpy(),
        rtol=1e-10,
    )


def test_idio_store_normalize_rank_in_unit_interval() -> None:
    panel = _make_panel(n_days=40, tickers=["AAA", "BBB", "CCC"])
    mkt = _idio_market_frame(panel)
    out = add_idiosyncratic_vol(panel, mkt, windows=20, normalize=True)
    vals = out["idio_vol"].dropna()
    assert len(vals) > 0
    assert vals.between(0.0, 1.0).all()


def test_idio_store_normalize_false_matches_raw() -> None:
    from data.processing.feature_implementation.idiosyncratic_vol import (
        add_idiosyncratic_vol as add_idiosyncratic_vol_raw,
    )

    panel = _make_panel(n_days=40, tickers=["AAA", "BBB"])
    mkt = _idio_market_frame(panel)
    store = add_idiosyncratic_vol(panel, mkt, windows=20, normalize=False)
    raw = add_idiosyncratic_vol_raw(panel, mkt, windows=20)
    both = store["idio_vol"].notna() & raw["idio_vol"].notna()
    np.testing.assert_allclose(
        store.loc[both, "idio_vol"].to_numpy(),
        raw.loc[both, "idio_vol"].to_numpy(),
        rtol=1e-10,
    )


def test_idio_store_no_lookahead_prefix_stability() -> None:
    panel = _make_panel(n_days=50, tickers=["AAA", "BBB"])
    mkt = _idio_market_frame(panel)
    full = add_idiosyncratic_vol(panel, mkt, windows=20, normalize=False)

    cutoff = panel["date"].sort_values().unique()[35]
    prefix = panel.loc[panel["date"] <= cutoff].copy()
    mkt_prefix = mkt.loc[mkt["date"] <= cutoff].copy()
    partial = add_idiosyncratic_vol(prefix, mkt_prefix, windows=20, normalize=False)

    merged = partial[["date", "ticker", "idio_vol"]].merge(
        full[["date", "ticker", "idio_vol"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    both = merged["idio_vol_partial"].notna() & merged["idio_vol_full"].notna()
    np.testing.assert_allclose(
        merged.loc[both, "idio_vol_partial"].to_numpy(),
        merged.loc[both, "idio_vol_full"].to_numpy(),
        rtol=1e-10,
    )


def test_idio_store_invalid_inputs() -> None:
    panel = _make_panel(n_days=30)
    mkt = _idio_market_frame(panel)
    with pytest.raises(ValueError, match="missing columns"):
        add_idiosyncratic_vol(panel.drop(columns=["close"]), mkt)
    with pytest.raises(ValueError, match="market_returns missing column"):
        add_idiosyncratic_vol(panel, mkt.drop(columns=["market_log_ret"]))
    with pytest.raises(ValueError, match="market_returns missing column"):
        add_idiosyncratic_vol(panel, mkt.drop(columns=["date"]))


# ---------------------------------------------------------------------------
# H-004 beta features
# ---------------------------------------------------------------------------

from data.processing.feature_implementation.beta import (
    blume_adjust,
    residual_momentum_signal,
    rolling_conditional_ols_stats,
    rolling_multi_ols_stats,
    rolling_ols_stats,
)
from data.processing.feature_implementation.beta_features import (
    _ensure_ff_workspace,
    _ensure_spy_workspace,
    _ws_col,
    drop_beta_workspace,
    parse_beta_factor_name,
)
from data.processing.feature_store import (
    add_beta,
    add_blume_beta,
    add_downside_beta,
    add_net_beta_spread,
    add_relative_downside_beta,
    add_relative_upside_beta,
    add_residual_momentum,
    add_upside_beta,
)


def _make_beta_panel(n_days: int = 80, tickers: list[str] | None = None) -> pd.DataFrame:
    """Larger synthetic panel for beta tests (need enough bars for windows)."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    frames = []
    for i, ticker in enumerate(tickers):
        base = 100.0 + i * 20.0
        returns = rng.normal(0.0005 * (i + 1), 0.015, n_days)
        close = base * np.exp(np.cumsum(returns))
        frames.append(
            pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "open": close * (1 - rng.uniform(0, 0.005, n_days)),
                "high": close * (1 + rng.uniform(0.001, 0.01, n_days)),
                "low": close * (1 - rng.uniform(0.001, 0.01, n_days)),
                "close": close,
                "volume": rng.uniform(1e5, 1e6, n_days),
            })
        )
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def _make_market_returns(panel: pd.DataFrame) -> pd.DataFrame:
    from data.processing.feature_implementation.beta import market_return_frame
    spy = panel[panel["ticker"] == panel["ticker"].iloc[0]][["date", "close"]].copy()
    spy["ticker"] = "SPY"
    return market_return_frame(spy)


def _make_ff_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Synthetic FF factors aligned to the panel dates."""
    rng = np.random.default_rng(99)
    dates = sorted(panel["date"].unique())
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "mkt_rf": rng.normal(0.0004, 0.01, n),
        "smb": rng.normal(0.0001, 0.005, n),
        "hml": rng.normal(0.0001, 0.005, n),
        "mom": rng.normal(0.0002, 0.008, n),
        "rf": np.full(n, 0.0002),
    })


# --- Primitive hand-checks ---

def test_rolling_conditional_ols_stats_basic() -> None:
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(0, 0.01, 120))
    y = 0.001 + 1.2 * x + rng.normal(0, 0.003, 120)
    down = rolling_conditional_ols_stats(y, x, 60, side="down", min_obs=10)
    up = rolling_conditional_ols_stats(y, x, 60, side="up", min_obs=10)
    assert not np.isnan(down["beta"].iloc[-1])
    assert not np.isnan(up["beta"].iloc[-1])
    assert down["n_obs"].iloc[-1] > 0
    assert up["n_obs"].iloc[-1] > 0
    assert down["beta"].iloc[:59].isna().all()
    with pytest.raises(ValueError, match="side"):
        rolling_conditional_ols_stats(y, x, 60, side="middle")


def test_rolling_multi_ols_parity_with_univariate() -> None:
    rng = np.random.default_rng(11)
    x = pd.Series(rng.normal(0, 0.01, 50))
    y = 0.002 + 0.8 * x + rng.normal(0, 0.002, 50)
    uni = rolling_ols_stats(y, x, 30)
    X_df = pd.DataFrame({"x": x})
    multi = rolling_multi_ols_stats(y, X_df, 30)
    both = uni["beta"].notna() & multi["x"].notna()
    np.testing.assert_allclose(
        uni.loc[both, "beta"].to_numpy(),
        multi.loc[both, "x"].to_numpy(),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        uni.loc[both, "alpha"].to_numpy(),
        multi.loc[both, "alpha"].to_numpy(),
        rtol=1e-8,
    )


def test_blume_adjust_formula() -> None:
    beta = pd.Series([0.5, 1.0, 1.5, 2.0])
    adj = blume_adjust(beta)
    expected = 0.67 * beta + 0.33
    np.testing.assert_allclose(adj.to_numpy(), expected.to_numpy())


def test_residual_momentum_signal_formula() -> None:
    rng = np.random.default_rng(3)
    resid = pd.Series(rng.normal(0, 0.01, 60))
    sig = residual_momentum_signal(resid, formation_window=40, skip=5)
    # Hand check last bar
    window = resid.iloc[60 - 40:60 - 5].to_numpy()
    expected = window.mean() / np.std(window, ddof=1)
    assert sig.iloc[-1] == pytest.approx(expected, rel=1e-10)
    assert sig.iloc[:39].isna().all()
    with pytest.raises(ValueError, match="formation_window must be greater than skip"):
        residual_momentum_signal(resid, 5, 5)
    with pytest.raises(ValueError, match="skip must be >= 0"):
        residual_momentum_signal(resid, 40, -1)


# --- Workspace idempotency ---

def test_spy_workspace_idempotency() -> None:
    panel = _make_beta_panel(n_days=80)
    mkt = _make_market_returns(panel)
    ws1 = _ensure_spy_workspace(panel, mkt, windows=[30, 40])
    ws_cols_1 = [c for c in ws1.columns if c.startswith("_ws_")]
    ws2 = _ensure_spy_workspace(ws1, mkt, windows=[30, 40])
    ws_cols_2 = [c for c in ws2.columns if c.startswith("_ws_")]
    assert set(ws_cols_1) == set(ws_cols_2)
    for c in ws_cols_1:
        both = ws1[c].notna() & ws2[c].notna()
        np.testing.assert_allclose(
            ws1.loc[both, c].to_numpy(),
            ws2.loc[both, c].to_numpy(),
            rtol=1e-12,
        )


def test_ff_workspace_idempotency() -> None:
    panel = _make_beta_panel(n_days=80)
    ff = _make_ff_factors(panel)
    ws1 = _ensure_ff_workspace(panel, ff, windows=[30, 40])
    ws_cols_1 = [c for c in ws1.columns if c.startswith("_ws_")]
    ws2 = _ensure_ff_workspace(ws1, ff, windows=[30, 40])
    ws_cols_2 = [c for c in ws2.columns if c.startswith("_ws_")]
    assert set(ws_cols_1) == set(ws_cols_2)
    for c in ws_cols_1:
        both = ws1[c].notna() & ws2[c].notna()
        np.testing.assert_allclose(
            ws1.loc[both, c].to_numpy(),
            ws2.loc[both, c].to_numpy(),
            rtol=1e-12,
        )


# --- Store callers: relationship identities (normalize=False) ---

def test_beta_store_spy_column_naming() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    single = add_beta(panel, mkt, benchmark="spy", windows=30, normalize=False)
    assert "beta" in single.columns
    assert "beta_30" not in single.columns

    multi = add_beta(panel, mkt, benchmark="spy", windows=[30, 40], normalize=False)
    assert "beta_30" in multi.columns
    assert "beta_40" in multi.columns
    assert "beta" not in multi.columns


def test_beta_store_ff_column_naming() -> None:
    panel = _make_beta_panel(n_days=60)
    ff = _make_ff_factors(panel)
    single = add_beta(panel, ff, benchmark="ff", windows=30, normalize=False)
    assert "smart_beta_smb" in single.columns
    assert "smart_beta_hml" in single.columns
    assert "smart_beta_mom" in single.columns

    multi = add_beta(panel, ff, benchmark="ff", windows=[30, 40], normalize=False)
    assert "smart_beta_smb_30" in multi.columns
    assert "smart_beta_hml_40" in multi.columns


def test_net_beta_spread_identity() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    result = add_upside_beta(panel, mkt, windows=30, normalize=False)
    result = add_downside_beta(result, mkt, windows=30, normalize=False)
    result = add_net_beta_spread(result, mkt, windows=30, normalize=False)
    both = result["net_beta_spread"].notna()
    np.testing.assert_allclose(
        result.loc[both, "net_beta_spread"].to_numpy(),
        (result.loc[both, "upside_beta"] - result.loc[both, "downside_beta"]).to_numpy(),
        rtol=1e-10,
    )


def test_relative_beta_identities() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    result = add_beta(panel, mkt, benchmark="spy", windows=30, normalize=False)
    result = add_downside_beta(result, mkt, windows=30, normalize=False)
    result = add_upside_beta(result, mkt, windows=30, normalize=False)
    result = add_relative_downside_beta(result, mkt, windows=30, normalize=False)
    result = add_relative_upside_beta(result, mkt, windows=30, normalize=False)

    both_down = result["rel_downside_beta"].notna() & result["downside_beta"].notna() & result["beta"].notna()
    np.testing.assert_allclose(
        result.loc[both_down, "rel_downside_beta"].to_numpy(),
        (result.loc[both_down, "downside_beta"] - result.loc[both_down, "beta"]).to_numpy(),
        rtol=1e-10,
    )
    both_up = result["rel_upside_beta"].notna() & result["upside_beta"].notna() & result["beta"].notna()
    np.testing.assert_allclose(
        result.loc[both_up, "rel_upside_beta"].to_numpy(),
        (result.loc[both_up, "upside_beta"] - result.loc[both_up, "beta"]).to_numpy(),
        rtol=1e-10,
    )


def test_blume_beta_identity() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    result = add_beta(panel, mkt, benchmark="spy", windows=30, normalize=False)
    result = add_blume_beta(result, mkt, windows=30, normalize=False)
    both = result["blume_beta"].notna() & result["beta"].notna()
    np.testing.assert_allclose(
        result.loc[both, "blume_beta"].to_numpy(),
        (0.67 * result.loc[both, "beta"] + 0.33).to_numpy(),
        rtol=1e-10,
    )


# --- normalize=True bounds ---

def test_beta_normalize_bounds() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    result = add_beta(panel, mkt, benchmark="spy", windows=30, normalize=True)
    vals = result["beta"].dropna()
    assert len(vals) > 0
    assert vals.between(0.0, 1.0).all()


# --- Invalid inputs ---

def test_beta_invalid_benchmark() -> None:
    panel = _make_beta_panel(n_days=40)
    mkt = _make_market_returns(panel)
    with pytest.raises(ValueError, match="benchmark"):
        add_beta(panel, mkt, benchmark="bad")


def test_residual_momentum_invalid_formation_skip() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    with pytest.raises(ValueError, match="formation_window must be > skip"):
        add_residual_momentum(panel, mkt, benchmark="spy", formation_window=10, skip=10)


# --- No-lookahead prefix parity ---

def test_beta_no_lookahead() -> None:
    panel = _make_beta_panel(n_days=80)
    mkt = _make_market_returns(panel)
    full = add_beta(panel, mkt, benchmark="spy", windows=30, normalize=False)
    cutoff = panel["date"].sort_values().unique()[50]
    prefix = panel[panel["date"] <= cutoff].copy()
    mkt_prefix = mkt[mkt["date"] <= cutoff].copy()
    partial = add_beta(prefix, mkt_prefix, benchmark="spy", windows=30, normalize=False)
    merged = partial[["date", "ticker", "beta"]].merge(
        full[["date", "ticker", "beta"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    both = merged["beta_partial"].notna() & merged["beta_full"].notna()
    np.testing.assert_allclose(
        merged.loc[both, "beta_partial"].to_numpy(),
        merged.loc[both, "beta_full"].to_numpy(),
        rtol=1e-10,
    )


# --- Multi-window parity ---

def test_beta_multi_window_parity() -> None:
    panel = _make_beta_panel(n_days=80)
    mkt = _make_market_returns(panel)
    multi = add_beta(panel, mkt, benchmark="spy", windows=[30, 40], normalize=False)
    single = add_beta(panel, mkt, benchmark="spy", windows=30, normalize=False)
    both = multi["beta_30"].notna() & single["beta"].notna()
    np.testing.assert_allclose(
        multi.loc[both, "beta_30"].to_numpy(),
        single.loc[both, "beta"].to_numpy(),
        rtol=1e-10,
    )


# --- Residual momentum store ---

def test_residual_momentum_spy_columns() -> None:
    panel = _make_beta_panel(n_days=80)
    mkt = _make_market_returns(panel)
    result = add_residual_momentum(
        panel, mkt, benchmark="spy",
        formation_window=[30, 40], skip=[5, 10],
        normalize=False,
    )
    assert "residual_mom_30_5" in result.columns
    assert "residual_mom_30_10" in result.columns
    assert "residual_mom_40_5" in result.columns
    assert "residual_mom_40_10" in result.columns

    single = add_residual_momentum(
        panel, mkt, benchmark="spy",
        formation_window=30, skip=5,
        normalize=False,
    )
    assert "residual_mom" in single.columns


def test_residual_momentum_ff_columns() -> None:
    panel = _make_beta_panel(n_days=80)
    ff = _make_ff_factors(panel)
    result = add_residual_momentum(
        panel, ff, benchmark="ff",
        formation_window=30, skip=5,
        normalize=False,
    )
    assert "smart_residual_mom" in result.columns


# --- parse_beta_factor_name round-trip ---

def test_parse_beta_factor_name_roundtrip() -> None:
    cases = [
        ("beta", {"feature": "beta", "window": None}),
        ("beta_252", {"feature": "beta", "window": 252}),
        ("downside_beta_126", {"feature": "downside_beta", "window": 126}),
        ("upside_beta_60", {"feature": "upside_beta", "window": 60}),
        ("net_beta_spread_252", {"feature": "net_beta_spread", "window": 252}),
        ("rel_downside_beta_126", {"feature": "rel_downside_beta", "window": 126}),
        ("rel_upside_beta_252", {"feature": "rel_upside_beta", "window": 252}),
        ("blume_beta_60", {"feature": "blume_beta", "window": 60}),
        ("smart_beta_smb_252", {"feature": "smart_beta_smb", "window": 252}),
        ("smart_beta_hml_126", {"feature": "smart_beta_hml", "window": 126}),
        ("smart_beta_mom_60", {"feature": "smart_beta_mom", "window": 60}),
        ("residual_mom_252_21", {"feature": "residual_mom", "K": 252, "S": 21}),
        ("smart_residual_mom_126_63", {"feature": "smart_residual_mom", "K": 126, "S": 63}),
        ("residual_mom", {"feature": "residual_mom", "K": None, "S": None}),
    ]
    for col, expected in cases:
        result = parse_beta_factor_name(col)
        assert result == expected, f"Failed for {col!r}: {result} != {expected}"

    assert parse_beta_factor_name("obv_mom_signed") is None
    assert parse_beta_factor_name("gk_vol_ratio_5_20") is None
    assert parse_beta_factor_name("random_column") is None


# --- drop_beta_workspace ---

def test_drop_beta_workspace() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    ws = _ensure_spy_workspace(panel, mkt, windows=[30])
    assert any(c.startswith("_ws_") for c in ws.columns)
    cleaned = drop_beta_workspace(ws)
    assert not any(c.startswith("_ws_") for c in cleaned.columns)


# --- FF fetcher schema (ETF Tier A, monkeypatched) ---

def test_ff_fetcher_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: str) -> None:
    import data.ingestion.alternative_data.fama_french_fetcher as ff_impl
    import data.ingestion.fama_french_fetcher as ff_shim

    # Two dates of closes per ETF → one simple-return row after dropna
    closes = {
        "SPY": (100.0, 101.0),
        "IWM": (50.0, 50.5),
        "IWD": (80.0, 80.8),
        "IWF": (90.0, 90.45),
        "MTUM": (70.0, 71.4),
        "BIL": (100.0, 100.02),
    }
    dates = [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")]
    call_count = {"n": 0}

    def mock_fetch_ohlcv(ticker, start_date, end_date=None, *, cache_dir=None):
        call_count["n"] += 1
        c0, c1 = closes[ticker.strip().upper()]
        return pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker.strip().upper(),
                "open": [c0, c1],
                "high": [c0, c1],
                "low": [c0, c1],
                "close": [c0, c1],
                "volume": [1_000_000, 1_000_000],
            }
        )

    monkeypatch.setattr(ff_impl, "fetch_ohlcv", mock_fetch_ohlcv)

    import tempfile
    cache_dir = tempfile.mkdtemp()
    # Call via compatibility shim (old import path)
    result = ff_shim.fetch_ff_factors_daily(cache_dir=cache_dir)
    assert set(result.columns) == {"date", "mkt_rf", "smb", "hml", "rf", "mom"}
    assert len(result) == 1

    spy_r = 101.0 / 100.0 - 1.0
    iwm_r = 50.5 / 50.0 - 1.0
    iwd_r = 80.8 / 80.0 - 1.0
    iwf_r = 90.45 / 90.0 - 1.0
    mtum_r = 71.4 / 70.0 - 1.0
    bil_r = 100.02 / 100.0 - 1.0

    assert result["rf"].iloc[0] == pytest.approx(bil_r)
    assert result["mkt_rf"].iloc[0] == pytest.approx(spy_r - bil_r)
    assert result["smb"].iloc[0] == pytest.approx(iwm_r - spy_r)
    assert result["hml"].iloc[0] == pytest.approx(iwd_r - iwf_r)
    assert result["mom"].iloc[0] == pytest.approx(mtum_r - spy_r)
    assert result["date"].is_monotonic_increasing

    n_after_build = call_count["n"]
    assert n_after_build == 6  # one fetch_ohlcv per ETF

    # Second call hits cache
    result2 = ff_shim.fetch_ff_factors_daily(cache_dir=cache_dir)
    assert call_count["n"] == n_after_build
    assert len(result2) == 1

