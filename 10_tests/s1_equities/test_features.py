"""Unit tests for feature-implementation modules and feature-store entrypoints."""

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
from data.processing.feature_implementation.momentum import (
    add_max_lottery_raw,
    add_near_52w_raw,
    add_raw_momentum,
    apply_near_52w_mode,
    max_lottery,
    near_52w_high,
    raw_momentum,
)
from data.processing.feature_implementation.obv_momentum import (
    add_obv,
    add_obv_trend,
    combine_momentum_obv,
    on_balance_volume,
    signs_agree,
)
from data.processing.feature_implementation.utilities import daily_simple_return
from data.processing.feature_store import (
    add_gk_vol_factors,
    add_idio_vol_factors,
    add_max_lottery_factors,
    add_near_52w_factors,
    add_obv_momentum_factors,
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
    signed = add_obv_momentum_factors(
        panel, lookback=5, skip=1, obv_window=3, feature_subset=["signed"]
    )
    strict = add_obv_momentum_factors(
        signed, lookback=5, skip=1, obv_window=3, feature_subset=["strict_zero"]
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


def test_obv_store_writes_raw_combined_signal() -> None:
    """H-001 has no normalize kwarg — stored values match the raw combined signal."""
    import inspect

    from data.processing.feature_implementation.obv_momentum import (
        add_obv_confirmed_combined,
    )

    sig = inspect.signature(add_obv_momentum_factors)
    assert "normalize" not in sig.parameters

    panel = _make_panel(n_days=30, tickers=["AAA", "BBB", "CCC"])
    out = add_obv_momentum_factors(
        panel, lookback=5, skip=1, obv_window=3, feature_subset=["signed"]
    )
    raw = add_obv_confirmed_combined(
        panel, lookback=5, skip=1, obv_window=3, mode="signed", col="_raw"
    )
    both = out["obv_mom_signed"].notna() & raw["_raw"].notna()
    np.testing.assert_allclose(
        out.loc[both, "obv_mom_signed"].to_numpy(),
        raw.loc[both, "_raw"].to_numpy(),
        rtol=1e-10,
    )


def test_no_lookahead_prefix_stability() -> None:
    panel = _make_panel(n_days=35)
    full = add_obv_momentum_factors(
        panel, lookback=5, skip=1, obv_window=3, feature_subset=["signed"]
    )
    cutoff = panel["date"].sort_values().unique()[-5]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_obv_momentum_factors(
        truncated, lookback=5, skip=1, obv_window=3, feature_subset=["signed"]
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
    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_obv_momentum_factors(panel, feature_subset=["nope"])
    with pytest.raises(ValueError, match="missing columns"):
        add_obv_momentum_factors(panel.drop(columns=["volume"]))


def test_obv_multi_window_column_names_and_parity() -> None:
    from data.processing.feature_implementation.utilities import windowed_column_name

    assert windowed_column_name("obv_mom_signed", 5, 1, 3, multi=False) == "obv_mom_signed"
    assert (
        windowed_column_name("obv_mom_signed", 5, 1, 3, multi=True) == "obv_mom_signed_5_1_3"
    )

    panel = _make_panel(n_days=40)
    multi = add_obv_momentum_factors(
        panel,
        lookback=[5, 8],
        skip=1,
        obv_window=3,
        feature_subset=["signed"],
    )
    assert "obv_mom_signed_5_1_3" in multi.columns
    assert "obv_mom_signed_8_1_3" in multi.columns
    assert "obv_mom_signed" not in multi.columns

    single_5 = add_obv_momentum_factors(
        panel, lookback=5, skip=1, obv_window=3, feature_subset=["signed"]
    )
    both = multi["obv_mom_signed_5_1_3"].notna() & single_5["obv_mom_signed"].notna()
    np.testing.assert_allclose(
        multi.loc[both, "obv_mom_signed_5_1_3"].to_numpy(),
        single_5.loc[both, "obv_mom_signed"].to_numpy(),
        rtol=1e-10,
    )

    mixed = add_obv_momentum_factors(
        panel,
        lookback=[5, 10],
        skip=[5],
        obv_window=3,
        feature_subset=["signed"],
    )
    assert "obv_mom_signed" in mixed.columns  # sole valid combo (10, 5, 3)
    assert "obv_mom_signed_5_5_3" not in mixed.columns
    assert "obv_mom_signed_10_5_3" not in mixed.columns

    with pytest.raises(ValueError, match="lookback must be greater than skip"):
        add_obv_momentum_factors(panel, lookback=[5], skip=[5], obv_window=3)


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
    ratio = add_gk_vol_factors(
        panel, gk_window=3, realised_window=5, feature_subset=["ratio"], normalize=False
    )
    log_r = add_gk_vol_factors(
        ratio, gk_window=3, realised_window=5, feature_subset=["log_ratio"], normalize=False
    )
    rev = add_gk_vol_factors(
        log_r, gk_window=3, realised_window=5, feature_subset=["reversal"], normalize=False
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


def test_gk_normalize_cs_zscore_finite() -> None:
    panel = _make_panel(n_days=35, tickers=["AAA", "BBB", "CCC"])
    out = add_gk_vol_factors(
        panel, gk_window=3, realised_window=5, feature_subset=["ratio"], normalize=True
    )
    vals = out["gk_vol_ratio"].dropna()
    assert len(vals) > 0
    assert np.isfinite(vals).all()
    # CS z is not confined to [0, 1] like pct-rank
    assert not vals.between(0.0, 1.0).all()
    # Within a date with 3 names, mean of finite z ≈ 0
    late = out["date"].max()
    day = out.loc[out["date"] == late, "gk_vol_ratio"].dropna()
    if len(day) >= 2:
        assert day.mean() == pytest.approx(0.0, abs=1e-10)


def test_gk_no_lookahead_prefix_stability() -> None:
    panel = _make_panel(n_days=40)
    full = add_gk_vol_factors(
        panel, gk_window=3, realised_window=5, feature_subset=["ratio"], normalize=False
    )
    cutoff = panel["date"].sort_values().unique()[-5]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_gk_vol_factors(
        truncated, gk_window=3, realised_window=5, feature_subset=["ratio"], normalize=False
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
    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_gk_vol_factors(panel, feature_subset=["nope"])
    with pytest.raises(ValueError, match="missing columns"):
        add_gk_vol_factors(panel.drop(columns=["high"]))


def test_gk_multi_window_column_names_and_parity() -> None:
    panel = _make_panel(n_days=45)
    multi = add_gk_vol_factors(
        panel,
        gk_window=[3, 4],
        realised_window=5,
        feature_subset=["ratio"],
        normalize=False,
    )
    assert "gk_vol_ratio_3_5" in multi.columns
    assert "gk_vol_ratio_4_5" in multi.columns
    assert "gk_vol_ratio" not in multi.columns

    single = add_gk_vol_factors(
        panel, gk_window=3, realised_window=5, feature_subset=["ratio"], normalize=False
    )
    both = multi["gk_vol_ratio_3_5"].notna() & single["gk_vol_ratio"].notna()
    np.testing.assert_allclose(
        multi.loc[both, "gk_vol_ratio_3_5"].to_numpy(),
        single.loc[both, "gk_vol_ratio"].to_numpy(),
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# H-006 52-week high proximity
# ---------------------------------------------------------------------------


def test_near_52w_high_series_formula() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 9.0])
    high = pd.Series([10.5, 11.5, 13.0, 12.0])
    out = near_52w_high(close, high, window=3)
    assert pd.isna(out.iloc[1])  # need 3 bars
    # At index 2: Hmax = max(10.5, 11.5, 13.0) = 13; close/Hmax = 12/13
    assert out.iloc[2] == pytest.approx(12.0 / 13.0)
    # At index 3: Hmax = max(11.5, 13.0, 12.0) = 13; close/Hmax = 9/13 (today included)
    assert out.iloc[3] == pytest.approx(9.0 / 13.0)


def test_apply_near_52w_modes() -> None:
    raw = pd.Series([1.0, 0.5, 0.0, -0.1])
    assert list(apply_near_52w_mode(raw, "ratio")) == pytest.approx([1.0, 0.5, 0.0, -0.1])
    log_m = apply_near_52w_mode(raw, "log_drawdown")
    assert log_m.iloc[0] == pytest.approx(0.0)
    assert log_m.iloc[1] == pytest.approx(np.log(0.5))
    assert pd.isna(log_m.iloc[2])
    assert pd.isna(log_m.iloc[3])


def test_near_52w_bad_inputs_nan() -> None:
    close = pd.Series([10.0, -1.0, 10.0])
    high = pd.Series([10.0, 10.0, 0.0])
    out = near_52w_high(close, high, window=1)
    assert out.iloc[0] == pytest.approx(1.0)
    assert pd.isna(out.iloc[1])  # non-positive close
    assert pd.isna(out.iloc[2])  # non-positive Hmax


def test_near_52w_panel_helper() -> None:
    panel = _make_panel(n_days=30)
    out = add_near_52w_raw(panel, window=5)
    assert "near_52w_raw" in out.columns
    assert out["near_52w_raw"].notna().any()


def test_near_52w_store_modes_and_column_names() -> None:
    panel = _make_panel(n_days=35)
    ratio = add_near_52w_factors(panel, window=5, feature_subset=["ratio"], normalize=False)
    log_d = add_near_52w_factors(ratio, window=5, feature_subset=["log_drawdown"], normalize=False)
    assert "near_52w_ratio" in log_d.columns
    assert "near_52w_log_drawdown" in log_d.columns

    pos = log_d["near_52w_ratio"] > 0
    np.testing.assert_allclose(
        log_d.loc[pos, "near_52w_log_drawdown"].to_numpy(),
        np.log(log_d.loc[pos, "near_52w_ratio"].to_numpy()),
    )


def test_near_52w_normalize_rank_in_unit_interval() -> None:
    panel = _make_panel(n_days=40)
    out = add_near_52w_factors(panel, window=5, feature_subset=["ratio"], normalize=True)
    vals = out["near_52w_ratio"].dropna()
    assert (vals >= 0).all() and (vals <= 1).all()


def test_near_52w_no_lookahead_prefix_stability() -> None:
    panel = _make_panel(n_days=50)
    full = add_near_52w_factors(panel, window=5, feature_subset=["ratio"], normalize=False)
    cutoff = panel["date"].sort_values().unique()[30]
    partial = add_near_52w_factors(
        panel.loc[panel["date"] <= cutoff].copy(),
        window=5,
        feature_subset=["ratio"],
        normalize=False,
    )
    merged = full.loc[full["date"] <= cutoff, ["date", "ticker", "near_52w_ratio"]].merge(
        partial[["date", "ticker", "near_52w_ratio"]],
        on=["date", "ticker"],
        suffixes=("_full", "_partial"),
    )
    both = merged["near_52w_ratio_full"].notna() & merged["near_52w_ratio_partial"].notna()
    np.testing.assert_allclose(
        merged.loc[both, "near_52w_ratio_full"].to_numpy(),
        merged.loc[both, "near_52w_ratio_partial"].to_numpy(),
        rtol=1e-10,
    )


def test_near_52w_invalid_mode_and_missing_columns() -> None:
    panel = _make_panel(n_days=25)
    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_near_52w_factors(panel, feature_subset=["nope"])
    with pytest.raises(ValueError, match="missing columns"):
        add_near_52w_factors(panel.drop(columns=["high"]))
    with pytest.raises(ValueError, match="window"):
        near_52w_high(panel["close"], panel["high"], window=0)


def test_near_52w_multi_window_column_names_and_parity() -> None:
    panel = _make_panel(n_days=45)
    multi = add_near_52w_factors(
        panel, window=[5, 10], feature_subset=["ratio"], normalize=False
    )
    assert "near_52w_ratio_5" in multi.columns
    assert "near_52w_ratio_10" in multi.columns
    assert "near_52w_ratio" not in multi.columns

    single = add_near_52w_factors(panel, window=5, feature_subset=["ratio"], normalize=False)
    both = multi["near_52w_ratio_5"].notna() & single["near_52w_ratio"].notna()
    np.testing.assert_allclose(
        multi.loc[both, "near_52w_ratio_5"].to_numpy(),
        single.loc[both, "near_52w_ratio"].to_numpy(),
        rtol=1e-10,
    )


# --- H-007 MAX (lottery demand) ---


def test_max_lottery_series_formula() -> None:
    # returns: idx 0 unused by first full window; window=3 ending at last bar
    rets = pd.Series([0.01, 0.05, -0.02, 0.03, 0.04])
    out = max_lottery(rets, n_extreme=2, window=3)
    # at index 3: window [-0.02, 0.03, 0.04]? wait indices: 1,2,3 = 0.05,-0.02,0.03 → top2 mean (0.05+0.03)/2
    # at index 4:  -0.02, 0.03, 0.04 → top2 (0.03+0.04)/2 = 0.035
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx((0.01 + 0.05) / 2)  # window [0.01, 0.05, -0.02]
    assert out.iloc[3] == pytest.approx((0.05 + 0.03) / 2)
    assert out.iloc[4] == pytest.approx((0.03 + 0.04) / 2)

    n1 = max_lottery(rets, n_extreme=1, window=3)
    assert n1.iloc[4] == pytest.approx(0.04)


def test_max_lottery_panel_helper() -> None:
    panel = _make_panel(n_days=30)
    out = add_max_lottery_raw(panel, n_extreme=2, window=5, mode="simple")
    assert "max_lottery_raw" in out.columns
    assert out["max_lottery_raw"].notna().any()


def test_max_lottery_store_modes_and_column_names() -> None:
    panel = _make_panel(n_days=30)
    simple = add_max_lottery_factors(
        panel, n_extreme=2, window=5, feature_subset=["simple"], normalize=False
    )
    log_m = add_max_lottery_factors(
        simple, n_extreme=2, window=5, feature_subset=["log"], normalize=False
    )
    assert "max_lottery_simple" in log_m.columns
    assert "max_lottery_log" in log_m.columns
    # log and simple extremes differ but both finite where defined
    both = log_m["max_lottery_simple"].notna() & log_m["max_lottery_log"].notna()
    assert both.any()
    # for small positive returns, log ret < simple ret so MAX(log) typically <= MAX(simple)
    assert (
        log_m.loc[both, "max_lottery_log"] <= log_m.loc[both, "max_lottery_simple"] + 1e-12
    ).all()


def test_max_lottery_bad_inputs_nan() -> None:
    close = pd.Series([100.0, np.nan, 102.0, 0.0, 103.0, 104.0])
    rets = daily_simple_return(close)
    assert np.isnan(rets.iloc[1])
    assert np.isnan(rets.iloc[3])  # non-positive close
    out = max_lottery(rets, n_extreme=2, window=3)
    # windows that lack 2 finite returns → NaN
    assert out.isna().any()


def test_max_lottery_normalize_cs_zscore_finite() -> None:
    panel = _make_panel(n_days=30, tickers=["AAA", "BBB", "CCC"])
    out = add_max_lottery_factors(
        panel, n_extreme=2, window=5, feature_subset=["simple"], normalize=True
    )
    vals = out["max_lottery_simple"].dropna()
    assert len(vals) > 0
    assert np.isfinite(vals).all()
    # CS z is not confined to [0, 1] like pct-rank
    assert not ((vals >= 0) & (vals <= 1)).all()
    late = out["date"].max()
    day = out.loc[out["date"] == late, "max_lottery_simple"].dropna()
    if len(day) >= 2:
        assert day.mean() == pytest.approx(0.0, abs=1e-10)


def test_max_lottery_no_lookahead_prefix_stability() -> None:
    panel = _make_panel(n_days=35)
    full = add_max_lottery_factors(
        panel, n_extreme=2, window=5, feature_subset=["simple"], normalize=False
    )
    cutoff = panel["date"].sort_values().unique()[20]
    partial = add_max_lottery_factors(
        panel.loc[panel["date"] <= cutoff].copy(),
        n_extreme=2,
        window=5,
        feature_subset=["simple"],
        normalize=False,
    )
    merged = full.loc[
        full["date"] <= cutoff, ["date", "ticker", "max_lottery_simple"]
    ].merge(
        partial[["date", "ticker", "max_lottery_simple"]],
        on=["date", "ticker"],
        suffixes=("_full", "_partial"),
    )
    both = (
        merged["max_lottery_simple_full"].notna()
        & merged["max_lottery_simple_partial"].notna()
    )
    np.testing.assert_allclose(
        merged.loc[both, "max_lottery_simple_full"].to_numpy(),
        merged.loc[both, "max_lottery_simple_partial"].to_numpy(),
        rtol=1e-10,
    )


def test_max_lottery_invalid_mode_and_missing_columns() -> None:
    panel = _make_panel(n_days=20)
    with pytest.raises(ValueError):
        add_max_lottery_factors(panel, feature_subset=["nope"])
    with pytest.raises(ValueError):
        add_max_lottery_factors(panel.drop(columns=["close"]))
    with pytest.raises(ValueError):
        add_max_lottery_factors(panel, n_extreme=5, window=3)
    with pytest.raises(ValueError):
        add_max_lottery_factors(panel, add_residuals=True)
    with pytest.raises(ValueError):
        max_lottery(panel["close"], n_extreme=0, window=5)


def test_max_lottery_multi_window_column_names_and_parity() -> None:
    panel = _make_panel(n_days=40)
    multi = add_max_lottery_factors(
        panel,
        n_extreme=[1, 2],
        window=[5, 8],
        feature_subset=["simple"],
        normalize=False,
    )
    assert "max_lottery_simple_1_5" in multi.columns
    assert "max_lottery_simple_2_8" in multi.columns
    assert "max_lottery_simple" not in multi.columns

    single = add_max_lottery_factors(
        panel, n_extreme=2, window=5, feature_subset=["simple"], normalize=False
    )
    both = multi["max_lottery_simple_2_5"].notna() & single["max_lottery_simple"].notna()
    np.testing.assert_allclose(
        multi.loc[both, "max_lottery_simple_2_5"].to_numpy(),
        single.loc[both, "max_lottery_simple"].to_numpy(),
        rtol=1e-10,
    )


def test_max_lottery_residuals_orthogonal_to_idio_rank() -> None:
    panel = _make_panel(n_days=35, tickers=["AAA", "BBB", "CCC", "DDD"])
    # Fake idio_vol correlated with but not identical to a return-scale proxy
    panel = panel.copy()
    panel["idio_vol"] = (
        panel.groupby("ticker", sort=False)["close"].pct_change().abs().fillna(0.01)
        + panel["ticker"].map({"AAA": 0.01, "BBB": 0.02, "CCC": 0.03, "DDD": 0.04})
    )
    out = add_max_lottery_factors(
        panel,
        n_extreme=2,
        window=5,
        feature_subset=["simple"],
        normalize=True,
        add_residuals=True,
        idio_vol_col="idio_vol",
    )
    assert "max_lottery_simple" in out.columns
    assert "max_lottery_simple_resid" in out.columns

    from data.processing.feature_implementation.utilities import (
        cross_sectional_pct_rank,
    )

    # On a late date with all names finite, resid should be ~orthogonal to idio rank
    late = out["date"].max()
    day = out.loc[out["date"] == late].copy()
    day["_idio_rank"] = cross_sectional_pct_rank(day, "idio_vol")
    mask = day["max_lottery_simple_resid"].notna() & day["_idio_rank"].notna()
    if mask.sum() >= 3:
        corr = np.corrcoef(
            day.loc[mask, "max_lottery_simple_resid"].to_numpy(),
            day.loc[mask, "_idio_rank"].to_numpy(),
        )[0, 1]
        assert abs(corr) < 1e-8


# --- H-003 / beta regression primitives ---


def test_beta_rolling_ols_hand_check() -> None:
    from data.processing.feature_implementation.linear_regression import rolling_ols_stats

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
    from data.processing.feature_implementation.beta_features import (
        add_rolling_beta,
        market_return_frame,
    )
    from data.processing.feature_implementation.utilities import regression_column_name

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
    from data.processing.feature_implementation.beta_features import market_return_frame
    from data.processing.feature_implementation.linear_regression import rolling_ols_stats
    from data.processing.feature_implementation.idiosyncratic_vol import add_idiosyncratic_vol

    panel = _make_panel(n_days=35, tickers=["AAA", "BBB"])
    spy = panel[panel["ticker"] == "AAA"][["date", "close"]].copy()
    spy["ticker"] = "SPY"
    mkt = market_return_frame(spy)

    out = add_idio_vol_factors(
        panel, mkt, windows=20, normalize=False, feature_subset=["idio_vol"]
    )
    assert "idio_vol" in out.columns
    assert "idio_vol_20" not in out.columns

    grp = out[out["ticker"] == "AAA"].sort_values("date")
    from data.processing.feature_implementation.utilities import log_return

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
    from data.processing.feature_implementation.beta_features import market_return_frame

    spy = panel[panel["ticker"] == panel["ticker"].iloc[0]][["date", "close"]].copy()
    spy["ticker"] = "SPY"
    return market_return_frame(spy)


def test_idio_store_single_vs_multi_window_parity() -> None:
    panel = _make_panel(n_days=40, tickers=["AAA", "BBB", "CCC"])
    mkt = _idio_market_frame(panel)

    multi = add_idio_vol_factors(panel, mkt, windows=[15, 20], normalize=False, feature_subset=["idio_vol"])
    assert "idio_vol_15" in multi.columns
    assert "idio_vol_20" in multi.columns
    assert "idio_vol" not in multi.columns

    single = add_idio_vol_factors(panel, mkt, windows=20, normalize=False, feature_subset=["idio_vol"])
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
    out = add_idio_vol_factors(panel, mkt, windows=20, normalize=True, feature_subset=["idio_vol"])
    vals = out["idio_vol"].dropna()
    assert len(vals) > 0
    assert vals.between(0.0, 1.0).all()


def test_idio_store_normalize_false_matches_raw() -> None:
    from data.processing.feature_implementation.idiosyncratic_vol import (
        add_idiosyncratic_vol as add_idiosyncratic_vol_raw,
    )

    panel = _make_panel(n_days=40, tickers=["AAA", "BBB"])
    mkt = _idio_market_frame(panel)
    store = add_idio_vol_factors(panel, mkt, windows=20, normalize=False, feature_subset=["idio_vol"])
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
    full = add_idio_vol_factors(panel, mkt, windows=20, normalize=False, feature_subset=["idio_vol"])

    cutoff = panel["date"].sort_values().unique()[35]
    prefix = panel.loc[panel["date"] <= cutoff].copy()
    mkt_prefix = mkt.loc[mkt["date"] <= cutoff].copy()
    partial = add_idio_vol_factors(prefix, mkt_prefix, windows=20, normalize=False, feature_subset=["idio_vol"])

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
        add_idio_vol_factors(panel.drop(columns=["close"]), mkt, feature_subset=["idio_vol"])
    with pytest.raises(ValueError, match="market_returns missing column"):
        add_idio_vol_factors(panel, mkt.drop(columns=["market_log_ret"]), feature_subset=["idio_vol"])
    with pytest.raises(ValueError, match="market_returns missing column"):
        add_idio_vol_factors(panel, mkt.drop(columns=["date"]), feature_subset=["idio_vol"])


# ---------------------------------------------------------------------------
# H-004 beta features
# ---------------------------------------------------------------------------

from data.processing.feature_implementation.beta_features import (
    blume_adjust,
    residual_momentum_signal,
)
from data.processing.feature_implementation.linear_regression import (
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
from data.processing.feature_store import add_beta_factors


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
    from data.processing.feature_implementation.beta_features import market_return_frame
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
    single = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="spy", windows=30, normalize=False)
    assert "beta" in single.columns
    assert "beta_30" not in single.columns

    multi = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="spy", windows=[30, 40], normalize=False)
    assert "beta_30" in multi.columns
    assert "beta_40" in multi.columns
    assert "beta" not in multi.columns


def test_beta_store_ff_column_naming() -> None:
    panel = _make_beta_panel(n_days=60)
    ff = _make_ff_factors(panel)
    single = add_beta_factors(panel, ff_factors=ff, feature_subset=["smart_beta_smb", "smart_beta_hml", "smart_beta_mom"], windows=30, normalize=False)
    assert "smart_beta_smb" in single.columns
    assert "smart_beta_hml" in single.columns
    assert "smart_beta_mom" in single.columns

    multi = add_beta_factors(panel, ff_factors=ff, feature_subset=["smart_beta_smb", "smart_beta_hml", "smart_beta_mom"], windows=[30, 40], normalize=False)
    assert "smart_beta_smb_30" in multi.columns
    assert "smart_beta_hml_40" in multi.columns


def test_net_beta_spread_identity() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    result = add_beta_factors(panel, market_returns=mkt, feature_subset=["upside_beta"], windows=30, normalize=False)
    result = add_beta_factors(result, market_returns=mkt, feature_subset=["downside_beta"], windows=30, normalize=False)
    result = add_beta_factors(result, market_returns=mkt, feature_subset=["net_beta_spread"], windows=30, normalize=False)
    both = result["net_beta_spread"].notna()
    np.testing.assert_allclose(
        result.loc[both, "net_beta_spread"].to_numpy(),
        (result.loc[both, "upside_beta"] - result.loc[both, "downside_beta"]).to_numpy(),
        rtol=1e-10,
    )


def test_relative_beta_identities() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    result = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="spy", windows=30, normalize=False)
    result = add_beta_factors(result, market_returns=mkt, feature_subset=["downside_beta"], windows=30, normalize=False)
    result = add_beta_factors(result, market_returns=mkt, feature_subset=["upside_beta"], windows=30, normalize=False)
    result = add_beta_factors(result, market_returns=mkt, feature_subset=["rel_downside_beta"], windows=30, normalize=False)
    result = add_beta_factors(result, market_returns=mkt, feature_subset=["rel_upside_beta"], windows=30, normalize=False)

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
    result = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="spy", windows=30, normalize=False)
    result = add_beta_factors(result, market_returns=mkt, feature_subset=["blume_beta"], windows=30)
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
    result = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="spy", windows=30, normalize=True)
    vals = result["beta"].dropna()
    assert len(vals) > 0
    assert vals.between(0.0, 1.0).all()


# --- Invalid inputs ---

def test_beta_invalid_benchmark() -> None:
    panel = _make_beta_panel(n_days=40)
    mkt = _make_market_returns(panel)
    with pytest.raises(ValueError, match="benchmark"):
        add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="bad")
    with pytest.raises(ValueError, match="benchmark"):
        add_beta_factors(
            panel,
            market_returns=mkt,
            feature_subset=["residual_mom"],
            benchmark="bad",
            formation_window=20,
            skip=5,
        )


def test_beta_store_rsp_matches_spy_contract() -> None:
    """benchmark='rsp' shares the univariate workspace / column contract with 'spy'."""
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    spy = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="spy", windows=30, normalize=False)
    rsp = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="rsp", windows=30, normalize=False)
    assert "beta" in rsp.columns
    assert "beta_30" not in rsp.columns
    both = spy["beta"].notna() & rsp["beta"].notna()
    np.testing.assert_allclose(
        spy.loc[both, "beta"].to_numpy(),
        rsp.loc[both, "beta"].to_numpy(),
        rtol=1e-12,
    )

    spy_mom = add_beta_factors(panel, market_returns=mkt, feature_subset=["residual_mom"], formation_window=30, skip=5)
    rsp_mom = add_beta_factors(panel, market_returns=mkt, feature_subset=["residual_mom"], formation_window=30, skip=5)
    assert "residual_mom" in rsp_mom.columns
    assert "smart_residual_mom" not in rsp_mom.columns
    both_m = spy_mom["residual_mom"].notna() & rsp_mom["residual_mom"].notna()
    np.testing.assert_allclose(
        spy_mom.loc[both_m, "residual_mom"].to_numpy(),
        rsp_mom.loc[both_m, "residual_mom"].to_numpy(),
        rtol=1e-12,
    )


def test_residual_momentum_invalid_formation_skip() -> None:
    panel = _make_beta_panel(n_days=60)
    mkt = _make_market_returns(panel)
    with pytest.raises(ValueError, match="formation_window must be > skip"):
        add_beta_factors(panel, market_returns=mkt, feature_subset=["residual_mom"], formation_window=10, skip=10)


# --- No-lookahead prefix parity ---

def test_beta_no_lookahead() -> None:
    panel = _make_beta_panel(n_days=80)
    mkt = _make_market_returns(panel)
    full = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="spy", windows=30, normalize=False)
    cutoff = panel["date"].sort_values().unique()[50]
    prefix = panel[panel["date"] <= cutoff].copy()
    mkt_prefix = mkt[mkt["date"] <= cutoff].copy()
    partial = add_beta_factors(prefix, market_returns=mkt_prefix, feature_subset=["beta"], benchmark="spy", windows=30, normalize=False)
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
    multi = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="spy", windows=[30, 40], normalize=False)
    single = add_beta_factors(panel, market_returns=mkt, feature_subset=["beta"], benchmark="spy", windows=30, normalize=False)
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
    result = add_beta_factors(panel, market_returns=mkt, feature_subset=["residual_mom"], formation_window=[30, 40], skip=[5, 10])
    assert "residual_mom_30_5" in result.columns
    assert "residual_mom_30_10" in result.columns
    assert "residual_mom_40_5" in result.columns
    assert "residual_mom_40_10" in result.columns

    single = add_beta_factors(panel, market_returns=mkt, feature_subset=["residual_mom"], formation_window=30, skip=5)
    assert "residual_mom" in single.columns


def test_residual_momentum_ff_columns() -> None:
    panel = _make_beta_panel(n_days=80)
    ff = _make_ff_factors(panel)
    result = add_beta_factors(panel, ff_factors=ff, feature_subset=["smart_residual_mom"], formation_window=30, skip=5)
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
    call_count = {"batch": 0, "single": 0}

    def mock_download_ohlcv(tickers, start, end, *, cache_dir=None, cache_label=""):
        call_count["batch"] += 1
        frames = []
        for ticker in tickers:
            c0, c1 = closes[str(ticker).strip().upper()]
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": str(ticker).strip().upper(),
                        "open": [c0, c1],
                        "high": [c0, c1],
                        "low": [c0, c1],
                        "close": [c0, c1],
                        "volume": [1_000_000, 1_000_000],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def mock_fetch_ohlcv(ticker, start_date, end_date=None, *, cache_dir=None):
        call_count["single"] += 1
        raise AssertionError("single-ticker fallback should not run when batch succeeds")

    monkeypatch.setattr(ff_impl, "_download_ohlcv", mock_download_ohlcv)
    monkeypatch.setattr(ff_impl, "fetch_ohlcv", mock_fetch_ohlcv)

    import tempfile
    cache_dir = tempfile.mkdtemp()
    result = ff_impl.fetch_ff_factors_daily(cache_dir=cache_dir)
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

    assert call_count["batch"] == 1
    assert call_count["single"] == 0

    # Second call hits cache
    result2 = ff_impl.fetch_ff_factors_daily(cache_dir=cache_dir)
    assert call_count["batch"] == 1
    assert len(result2) == 1


# ---------------------------------------------------------------------------
# H-005 Size & Value features
# ---------------------------------------------------------------------------

from data.processing.feature_implementation.size_and_valuation_features import (
    book_yield,
    earnings_yield,
    log_market_cap,
    size_momentum,
    valuation_roc,
    value_momentum_distance,
)
from data.processing.feature_store import add_size_value_factors


def _make_sv_panel(
    n_days: int = 60,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Synthetic panel with market_cap, pe, pb for H-005 tests."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(55)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    frames = []
    for i, ticker in enumerate(tickers):
        base = 100.0 + i * 20.0
        returns = rng.normal(0.0005 * (i + 1), 0.012, n_days)
        close = base * np.exp(np.cumsum(returns))
        shares = 1e6 * (1 + i)
        market_cap = close * shares
        pe = 15.0 + rng.normal(0, 2, n_days)
        pb = 2.0 + rng.normal(0, 0.3, n_days)
        pe = np.clip(pe, 0.5, 50)
        pb = np.clip(pb, 0.2, 10)
        frames.append(
            pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "open": close * (1 - rng.uniform(0, 0.005, n_days)),
                "high": close * (1 + rng.uniform(0.001, 0.01, n_days)),
                "low": close * (1 - rng.uniform(0.001, 0.01, n_days)),
                "close": close,
                "volume": rng.uniform(1e5, 1e6, n_days),
                "market_cap": market_cap,
                "pe": pe,
                "pb": pb,
            })
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def test_book_yield_hand_check() -> None:
    pb = pd.Series([2.0, 0.5, -1.0, 0.0, 10.0])
    by = book_yield(pb)
    assert by.iloc[0] == pytest.approx(0.5)
    assert by.iloc[1] == pytest.approx(2.0)
    assert pd.isna(by.iloc[2])
    assert pd.isna(by.iloc[3])
    assert by.iloc[4] == pytest.approx(0.1)


def test_earnings_yield_hand_check() -> None:
    pe = pd.Series([20.0, 5.0, -2.0, 0.0])
    ey = earnings_yield(pe)
    assert ey.iloc[0] == pytest.approx(0.05)
    assert ey.iloc[1] == pytest.approx(0.2)
    assert pd.isna(ey.iloc[2])
    assert pd.isna(ey.iloc[3])


def test_log_market_cap_hand_check() -> None:
    mcap = pd.Series([1e6, 1e9, -1.0, 0.0])
    lm = log_market_cap(mcap)
    assert lm.iloc[0] == pytest.approx(np.log(1e6))
    assert lm.iloc[1] == pytest.approx(np.log(1e9))
    assert pd.isna(lm.iloc[2])
    assert pd.isna(lm.iloc[3])


def test_valuation_roc_hand_check() -> None:
    val = pd.Series([10.0, 12.0, 15.0, 18.0, 20.0])
    roc = valuation_roc(val, window=2)
    expected_at_2 = np.log(15.0) - np.log(10.0)
    expected_at_4 = np.log(20.0) - np.log(15.0)
    assert pd.isna(roc.iloc[0])
    assert pd.isna(roc.iloc[1])
    assert roc.iloc[2] == pytest.approx(expected_at_2)
    assert roc.iloc[4] == pytest.approx(expected_at_4)


def test_size_momentum_hand_check() -> None:
    mcap = pd.Series([1e6, 1.1e6, 1.2e6, 1.05e6])
    sm = size_momentum(mcap, window=2)
    expected_at_2 = np.log(1.2e6 / 1e6)
    expected_at_3 = np.log(1.05e6 / 1.1e6)
    assert pd.isna(sm.iloc[0])
    assert pd.isna(sm.iloc[1])
    assert sm.iloc[2] == pytest.approx(expected_at_2)
    assert sm.iloc[3] == pytest.approx(expected_at_3)


def test_value_momentum_distance_geometry() -> None:
    val_rank = pd.Series([1.0, 0.0, 0.5])
    mom_rank = pd.Series([1.0, 0.0, 0.5])
    dist = value_momentum_distance(val_rank, mom_rank)
    assert dist.iloc[0] == pytest.approx(0.0)
    assert dist.iloc[1] == pytest.approx(np.sqrt(2.0))
    assert dist.iloc[2] == pytest.approx(np.sqrt(0.5))


def test_sv_store_missing_columns() -> None:
    panel = _make_panel(n_days=20)
    with pytest.raises(ValueError, match="missing columns"):
        add_size_value_factors(
            panel, feature_subset=["book_yield"], size_value_data_exists=True
        )
    with pytest.raises(ValueError, match="missing columns"):
        add_size_value_factors(
            panel, feature_subset=["earnings_yield"], size_value_data_exists=True
        )
    with pytest.raises(ValueError, match="missing columns"):
        add_size_value_factors(
            panel, feature_subset=["log_mcap"], size_value_data_exists=True
        )
    with pytest.raises(ValueError, match="missing columns"):
        add_size_value_factors(
            panel, feature_subset=["size_mom"], size_value_data_exists=True
        )


def test_sv_store_invalid_metric() -> None:
    panel = _make_sv_panel(n_days=20)
    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_size_value_factors(panel, feature_subset=["val_roc_bad"], size_value_data_exists=True)


def test_sv_store_normalize_bounds() -> None:
    panel = _make_sv_panel(n_days=60)
    by_out = add_size_value_factors(panel, normalize=True, feature_subset=["book_yield"], size_value_data_exists=True)
    vals = by_out["book_yield"].dropna()
    assert len(vals) > 0
    assert vals.between(0.0, 1.0).all()

    ey_out = add_size_value_factors(panel, normalize=True, feature_subset=["earnings_yield"], size_value_data_exists=True)
    vals = ey_out["earnings_yield"].dropna()
    assert len(vals) > 0
    assert vals.between(0.0, 1.0).all()

    lm_out = add_size_value_factors(panel, normalize=True, feature_subset=["log_mcap"], size_value_data_exists=True)
    vals = lm_out["log_mcap"].dropna()
    assert len(vals) > 0
    assert vals.between(0.0, 1.0).all()


def test_sv_valuation_roc_column_names() -> None:
    panel = _make_sv_panel(n_days=60)
    single = add_size_value_factors(panel, window=5, feature_subset=["val_roc_pb"], size_value_data_exists=True)
    assert "val_roc_pb" in single.columns
    assert "val_roc_pb_5" not in single.columns

    multi = add_size_value_factors(panel, window=[5, 10], feature_subset=["val_roc_pb"], size_value_data_exists=True)
    assert "val_roc_pb_5" in multi.columns
    assert "val_roc_pb_10" in multi.columns
    assert "val_roc_pb" not in multi.columns


def test_sv_size_momentum_multi_window_parity() -> None:
    panel = _make_sv_panel(n_days=60)
    multi = add_size_value_factors(panel, window=[5, 10], feature_subset=["size_mom"], size_value_data_exists=True)
    single = add_size_value_factors(panel, window=5, feature_subset=["size_mom"], size_value_data_exists=True)
    both = multi["size_mom_5"].notna() & single["size_mom"].notna()
    np.testing.assert_allclose(
        multi.loc[both, "size_mom_5"].to_numpy(),
        single.loc[both, "size_mom"].to_numpy(),
        rtol=1e-10,
    )


def test_sv_no_lookahead_prefix_stability() -> None:
    panel = _make_sv_panel(n_days=60)
    full = add_size_value_factors(panel, normalize=False, feature_subset=["book_yield"], size_value_data_exists=True)
    cutoff = panel["date"].sort_values().unique()[-10]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_size_value_factors(truncated, normalize=False, feature_subset=["book_yield"], size_value_data_exists=True)
    merged = partial[["date", "ticker", "book_yield"]].merge(
        full[["date", "ticker", "book_yield"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    both = merged["book_yield_partial"].notna() & merged["book_yield_full"].notna()
    np.testing.assert_allclose(
        merged.loc[both, "book_yield_partial"].to_numpy(),
        merged.loc[both, "book_yield_full"].to_numpy(),
        rtol=1e-10,
    )


def test_sv_value_momentum_interaction_column() -> None:
    panel = _make_sv_panel(n_days=60)
    result = add_size_value_factors(
        panel, mom_lookback=5, mom_skip=1, feature_subset=["val_mom_interact"],
        size_value_data_exists=True)
    assert "val_mom_interact" in result.columns
    assert result["val_mom_interact"].notna().any()


def test_sv_value_momentum_distance_column() -> None:
    panel = _make_sv_panel(n_days=60)
    result = add_size_value_factors(
        panel, mom_lookback=5, mom_skip=1, feature_subset=["val_mom_dist"],
        size_value_data_exists=True)
    assert "val_mom_dist" in result.columns
    vals = result["val_mom_dist"].dropna()
    assert len(vals) > 0
    assert (vals >= 0).all()


def test_sv_value_momentum_residual_orthogonality() -> None:
    panel = _make_sv_panel(n_days=60)
    result = add_size_value_factors(
        panel,
        regression_window=20,
        mom_lookback=5,
        mom_skip=1,
        feature_subset=["val_mom_resid"],
        size_value_data_exists=True)
    assert "val_mom_resid" in result.columns
    assert result["val_mom_resid"].notna().any()


def test_sv_store_fetches_and_merges_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _make_panel(n_days=40, tickers=["AAA", "BBB"])
    panel = panel.copy()
    panel["feature_date"] = pd.to_datetime(panel["date"]) - pd.Timedelta(days=1)
    calls = {"n": 0}

    def _fake_fetch(tickers, start_date=None, end_date=None, **kwargs):
        calls["n"] += 1
        assert set(tickers) == {"AAA", "BBB"}
        rows = []
        for t in tickers:
            for fd in panel.loc[panel["ticker"] == t, "feature_date"].unique():
                rows.append(
                    {
                        "date": pd.Timestamp(fd),
                        "ticker": t,
                        "shares_outstanding": 1e6,
                        "book_equity": 1e8,
                        "eps_ttm": 2.0,
                        "market_cap": 1e9,
                        "pe": 15.0,
                        "pb": 2.0,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "data.ingestion.alternative_data.sec_companyfacts.fetch_size_value_daily",
        _fake_fetch,
    )
    out = add_size_value_factors(
        panel, feature_subset=["book_yield"], size_value_data_exists=False
    )
    assert calls["n"] == 1
    assert "pb" in out.columns
    assert "book_yield" in out.columns
    assert out["pb"].notna().all()

    out2 = add_size_value_factors(
        out, feature_subset=["log_mcap"], size_value_data_exists=True
    )
    assert calls["n"] == 1
    assert "log_mcap" in out2.columns


def test_sv_amihud_only_skips_sec_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(n_days=40)
    panel = panel.copy()
    panel["volume"] = panel["volume"] * (1.0 + 0.1 * np.arange(len(panel)) % 7)

    def _boom(*args, **kwargs):
        raise AssertionError("fetch_size_value_daily should not be called")

    monkeypatch.setattr(
        "data.ingestion.alternative_data.sec_companyfacts.fetch_size_value_daily",
        _boom,
    )
    out = add_size_value_factors(
        panel, feature_subset=["amihud"], amihud_window=5, normalize=False
    )
    assert "amihud" in out.columns


# ---------------------------------------------------------------------------
# H-008 Gross Profitability
# ---------------------------------------------------------------------------

from data.processing.feature_implementation.gross_profitability import (
    gross_profitability,
)
from data.processing.feature_store import add_gross_profitability_factors


def _make_gp_panel(
    n_days: int = 40,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Synthetic panel with ``gp_asset`` for H-008 tests."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(88)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    frames = []
    for i, ticker in enumerate(tickers):
        base = 100.0 + i * 20.0
        returns = rng.normal(0.0005 * (i + 1), 0.012, n_days)
        close = base * np.exp(np.cumsum(returns))
        gp_asset = 0.1 + 0.05 * i + rng.normal(0, 0.01, n_days)
        frames.append(
            pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "close": close,
                "gp_asset": gp_asset,
            })
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def test_gross_profitability_hand_check() -> None:
    gp = pd.Series([0.25, 0.5, np.nan, np.inf, -0.1])
    out = gross_profitability(gp)
    assert out.iloc[0] == pytest.approx(0.25)
    assert out.iloc[1] == pytest.approx(0.5)
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])
    assert out.iloc[4] == pytest.approx(-0.1)


def test_gp_store_column_name() -> None:
    panel = _make_gp_panel(n_days=20)
    result = add_gross_profitability_factors(panel, normalize=False, feature_subset=["gross_profitability"], gross_profitability_data_exists=True)
    assert "gross_profitability" in result.columns
    both = result["gross_profitability"].notna() & result["gp_asset"].notna()
    np.testing.assert_allclose(
        result.loc[both, "gross_profitability"].to_numpy(),
        result.loc[both, "gp_asset"].to_numpy(),
        rtol=1e-12,
    )


def test_gp_store_missing_columns() -> None:
    panel = _make_panel(n_days=20)
    with pytest.raises(ValueError, match="missing columns"):
        add_gross_profitability_factors(panel, feature_subset=["gross_profitability"], gross_profitability_data_exists=True)


def test_gp_store_normalize_bounds() -> None:
    panel = _make_gp_panel(n_days=40)
    out = add_gross_profitability_factors(panel, normalize=True, feature_subset=["gross_profitability"], gross_profitability_data_exists=True)
    vals = out["gross_profitability"].dropna()
    assert len(vals) > 0
    assert vals.between(0.0, 1.0).all()


def test_gp_no_lookahead_prefix_stability() -> None:
    panel = _make_gp_panel(n_days=40)
    full = add_gross_profitability_factors(panel, normalize=False, feature_subset=["gross_profitability"], gross_profitability_data_exists=True)
    cutoff = panel["date"].sort_values().unique()[-10]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_gross_profitability_factors(truncated, normalize=False, feature_subset=["gross_profitability"], gross_profitability_data_exists=True)
    merged = partial[["date", "ticker", "gross_profitability"]].merge(
        full[["date", "ticker", "gross_profitability"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    both = (
        merged["gross_profitability_partial"].notna()
        & merged["gross_profitability_full"].notna()
    )
    np.testing.assert_allclose(
        merged.loc[both, "gross_profitability_partial"].to_numpy(),
        merged.loc[both, "gross_profitability_full"].to_numpy(),
        rtol=1e-10,
    )


def test_gp_store_fetches_and_merges_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _make_panel(n_days=30, tickers=["AAA", "BBB"])
    panel = panel.copy()
    panel["feature_date"] = pd.to_datetime(panel["date"]) - pd.Timedelta(days=1)
    calls = {"n": 0}

    def _fake_fetch(tickers, start_date=None, end_date=None, **kwargs):
        calls["n"] += 1
        rows = []
        for t in tickers:
            for fd in panel.loc[panel["ticker"] == t, "feature_date"].unique():
                rows.append(
                    {
                        "date": pd.Timestamp(fd),
                        "ticker": t,
                        "gross_profit_ttm": 1e8,
                        "assets": 5e8,
                        "gp_asset": 0.2,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "data.ingestion.alternative_data.sec_companyfacts.fetch_gross_profitability_daily",
        _fake_fetch,
    )
    out = add_gross_profitability_factors(
        panel,
        feature_subset=["gross_profitability"],
        normalize=False,
        gross_profitability_data_exists=False,
    )
    assert calls["n"] == 1
    assert "gp_asset" in out.columns
    assert "gross_profitability" in out.columns
    assert out["gp_asset"].notna().all()

    out2 = add_gross_profitability_factors(
        out,
        feature_subset=["gross_profitability"],
        normalize=True,
        gross_profitability_data_exists=True,
    )
    assert calls["n"] == 1
    assert out2["gross_profitability"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# H-010 short-selling pressure
# ---------------------------------------------------------------------------

from data.processing.feature_implementation.filing_clock import (
    days_since_filing,
    expected_days_until_filing,
)
from data.processing.feature_implementation.short_flow import (
    abnormal_short_flow,
    short_volume_ratio,
)
from data.processing.feature_store import add_short_flow_factors


def _make_short_flow_panel(
    n_days: int = 80,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Synthetic panel with FINRA short-volume columns for H-010 tests."""
    if tickers is None:
        tickers = ["AAA", "BBB"]
    rng = np.random.default_rng(101)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    frames = []
    for i, ticker in enumerate(tickers):
        total = rng.uniform(1e5, 5e5, n_days)
        short = total * (0.3 + 0.1 * i + 0.05 * rng.normal(0, 1, n_days))
        short = np.clip(short, 0, total)
        exempt = short * 0.05
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "short_volume": short,
                    "short_exempt_volume": exempt,
                    "total_volume": total,
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def test_short_volume_ratio_hand_check() -> None:
    short = pd.Series([10.0, 20.0, 5.0, 1.0])
    total = pd.Series([100.0, 50.0, 0.0, -10.0])
    ratio = short_volume_ratio(short, total)
    assert ratio.iloc[0] == pytest.approx(0.1)
    assert ratio.iloc[1] == pytest.approx(0.4)
    assert pd.isna(ratio.iloc[2])
    assert pd.isna(ratio.iloc[3])


def test_abnormal_short_flow_hand_check_baseline_excludes_current() -> None:
    # smooth=2, baseline=2 on svr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # sm = [nan, 0.15, 0.25, 0.35, 0.45, 0.55]
    # lagged sm = [nan, nan, 0.15, 0.25, 0.35, 0.45]
    # at idx 3: base = [0.15, 0.25], mean=0.2, std(ddof=0)=0.05
    # z = (0.35 - 0.2) / 0.05 = 3.0  (current sm excluded from base)
    svr = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    z = abnormal_short_flow(svr, smooth_window=2, baseline_window=2)
    assert pd.isna(z.iloc[2])
    assert z.iloc[3] == pytest.approx(3.0)
    # at idx 4: base = [0.25, 0.35], mean=0.3, std=0.05; z=(0.45-0.3)/0.05=3.0
    assert z.iloc[4] == pytest.approx(3.0)


def test_abnormal_short_flow_zero_variance_nan() -> None:
    svr = pd.Series([0.25] * 20)
    z = abnormal_short_flow(svr, smooth_window=3, baseline_window=5)
    # Constant series → rolling std == 0 → NaN (not 0 or inf)
    assert z.isna().all()
    assert not np.isinf(z.to_numpy(dtype=float)).any()


def test_short_flow_store_modes_and_columns() -> None:
    panel = _make_short_flow_panel(n_days=80)
    abn = add_short_flow_factors(panel, smooth_window=5, baseline_window=20, feature_subset=["abnormal"], short_volume_data_exists=True)
    assert "short_flow_abnormal" in abn.columns
    assert abn["short_flow_abnormal"].notna().any()

    ratio = add_short_flow_factors(panel, feature_subset=["ratio"], short_volume_data_exists=True)
    assert "short_flow_ratio" in ratio.columns
    both = ratio["short_flow_ratio"].notna()
    expected = short_volume_ratio(panel["short_volume"], panel["total_volume"])
    np.testing.assert_allclose(
        ratio.loc[both, "short_flow_ratio"].to_numpy(),
        expected.loc[both].to_numpy(),
        rtol=1e-12,
    )

    exempt = add_short_flow_factors(panel, feature_subset=["exempt_ratio"], short_volume_data_exists=True)
    assert "short_flow_exempt_ratio" in exempt.columns
    assert exempt["short_flow_exempt_ratio"].notna().any()


def test_short_flow_multi_window_parity() -> None:
    panel = _make_short_flow_panel(n_days=80)
    multi = add_short_flow_factors(
        panel,
        smooth_window=[3, 5],
        baseline_window=[10, 20],
        feature_subset=["abnormal"],
        short_volume_data_exists=True)
    assert "short_flow_abnormal_3_10" in multi.columns
    assert "short_flow_abnormal_5_20" in multi.columns
    assert "short_flow_abnormal" not in multi.columns

    single = add_short_flow_factors(
        panel, smooth_window=5, baseline_window=20, feature_subset=["abnormal"],
        short_volume_data_exists=True)
    both = (
        multi["short_flow_abnormal_5_20"].notna()
        & single["short_flow_abnormal"].notna()
    )
    np.testing.assert_allclose(
        multi.loc[both, "short_flow_abnormal_5_20"].to_numpy(),
        single.loc[both, "short_flow_abnormal"].to_numpy(),
        rtol=1e-10,
    )


def test_short_flow_invalid_mode_and_missing_cols() -> None:
    panel = _make_short_flow_panel(n_days=30)
    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_short_flow_factors(panel, feature_subset=["bad"], short_volume_data_exists=True)
    bare = _make_panel(n_days=20)
    with pytest.raises(ValueError, match="missing columns"):
        add_short_flow_factors(bare, feature_subset=["abnormal"], short_volume_data_exists=True)
    with pytest.raises(ValueError, match="missing columns"):
        add_short_flow_factors(bare, feature_subset=["exempt_ratio"], short_volume_data_exists=True)


def test_short_flow_no_lookahead_prefix_stability() -> None:
    panel = _make_short_flow_panel(n_days=80)
    full = add_short_flow_factors(panel, smooth_window=5, baseline_window=20, feature_subset=["abnormal"], short_volume_data_exists=True)
    cutoff = panel["date"].sort_values().unique()[-15]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_short_flow_factors(
        truncated, smooth_window=5, baseline_window=20, feature_subset=["abnormal"],
        short_volume_data_exists=True)
    merged = partial[["date", "ticker", "short_flow_abnormal"]].merge(
        full[["date", "ticker", "short_flow_abnormal"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    both = (
        merged["short_flow_abnormal_partial"].notna()
        & merged["short_flow_abnormal_full"].notna()
    )
    np.testing.assert_allclose(
        merged.loc[both, "short_flow_abnormal_partial"].to_numpy(),
        merged.loc[both, "short_flow_abnormal_full"].to_numpy(),
        rtol=1e-10,
    )


def test_filing_clock_hand_checks_including_overdue() -> None:
    dates = pd.Series(pd.to_datetime(["2023-01-10", "2023-02-10", "2023-04-10"]))
    last_filed = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"]))
    since = days_since_filing(dates, last_filed)
    assert since.iloc[0] == pytest.approx(9.0)
    assert since.iloc[1] == pytest.approx(40.0)

    expected_next = pd.Series(
        pd.to_datetime(["2023-02-01", "2023-02-01", "2023-02-01"])
    )
    until = expected_days_until_filing(dates, expected_next)
    assert until.iloc[0] == pytest.approx(22.0)  # Jan 10 → Feb 1
    assert until.iloc[1] == pytest.approx(-9.0)  # overdue
    assert until.iloc[2] < 0


def test_filing_event_clock_store_modes() -> None:
    dates = pd.bdate_range("2023-01-02", periods=10)
    panel = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "last_filed": pd.Timestamp("2022-12-15"),
            "expected_next_filed": pd.Timestamp("2023-01-20"),
        }
    )
    since = add_short_flow_factors(panel, feature_subset=["filing_since"], filing_clock_data_exists=True)
    assert "filing_clock_since" in since.columns
    assert since["filing_clock_since"].iloc[0] == pytest.approx(
        (dates[0] - pd.Timestamp("2022-12-15")).days
    )

    until = add_short_flow_factors(panel, feature_subset=["filing_expected_until"], filing_clock_data_exists=True)
    assert "filing_clock_expected_until" in until.columns
    # After 2023-01-20 the signed days go negative (overdue)
    late = until[until["date"] > pd.Timestamp("2023-01-20")]
    assert (late["filing_clock_expected_until"] < 0).all()

    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_short_flow_factors(panel, feature_subset=["nope"])
    with pytest.raises(ValueError, match="missing columns"):
        add_short_flow_factors(
            pd.DataFrame({"date": dates, "ticker": "AAA"}),
            feature_subset=["filing_since"],
            filing_clock_data_exists=True,
        )

def test_expected_next_filed_pit_no_lookahead() -> None:
    """Forecast at event i uses only filings 0..i (primary: filed[i-3]+365)."""
    from data.ingestion.alternative_data.sec_companyfacts import (
        _extract_filing_clock_events,
        _forecast_expected_next_filed,
    )

    filed = [
        pd.Timestamp("2020-01-15"),
        pd.Timestamp("2020-04-15"),
        pd.Timestamp("2020-07-15"),
        pd.Timestamp("2020-10-15"),
        pd.Timestamp("2021-01-20"),
    ]
    # i=0 → NaT
    assert pd.isna(_forecast_expected_next_filed(filed, 0))
    # i=1 → median of 1 gap = 91 days (approx Jan→Apr)
    fb1 = _forecast_expected_next_filed(filed, 1)
    assert fb1 == filed[1] + pd.Timedelta(days=(filed[1] - filed[0]).days)
    # i=3 → primary: filed[0] + 365
    assert _forecast_expected_next_filed(filed, 3) == filed[0] + pd.Timedelta(
        days=365
    )
    # i=4 → primary: filed[1] + 365
    assert _forecast_expected_next_filed(filed, 4) == filed[1] + pd.Timedelta(
        days=365
    )

    facts = {
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "filed": ts.strftime("%Y-%m-%d"),
                                "val": 1.0,
                                "form": "10-Q",
                                "end": ts.strftime("%Y-%m-%d"),
                            }
                            for ts in filed
                        ]
                    }
                }
            }
        }
    }
    events = _extract_filing_clock_events(facts, ("10-Q", "10-K", "10-Q/A", "10-K/A"))
    assert len(events) == 5
    assert pd.isna(events["expected_next_filed"].iloc[0])
    assert events["expected_next_filed"].iloc[3] == filed[0] + pd.Timedelta(days=365)
    # Later forecasts must not equal a future filed date that wasn't available
    # at i=3 (filed[4] was unknown): primary uses filed[0]+365 only.
    assert events["expected_next_filed"].iloc[3] != filed[4]


def test_short_flow_store_fetches_finra_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _make_panel(n_days=40, tickers=["AAA", "BBB"])
    panel = panel.copy()
    panel["feature_date"] = pd.to_datetime(panel["date"]) - pd.Timedelta(days=1)
    calls = {"finra": 0, "filing": 0}

    def _fake_finra(tickers, start_date=None, end_date=None, **kwargs):
        calls["finra"] += 1
        rows = []
        for t in tickers:
            for fd in panel.loc[panel["ticker"] == t, "feature_date"].unique():
                rows.append(
                    {
                        "date": pd.Timestamp(fd),
                        "ticker": t,
                        "short_volume": 1e5,
                        "short_exempt_volume": 1e3,
                        "total_volume": 4e5,
                    }
                )
        return pd.DataFrame(rows)

    def _boom_filing(*args, **kwargs):
        calls["filing"] += 1
        raise AssertionError("filing clock should not be fetched for ratio-only")

    monkeypatch.setattr(
        "data.ingestion.alternative_data.finra_short_volume.fetch_short_volume_daily",
        _fake_finra,
    )
    monkeypatch.setattr(
        "data.ingestion.alternative_data.sec_companyfacts.fetch_filing_clock_daily",
        _boom_filing,
    )
    out = add_short_flow_factors(
        panel, feature_subset=["ratio"], short_volume_data_exists=False
    )
    assert calls["finra"] == 1
    assert calls["filing"] == 0
    assert "short_volume" in out.columns
    assert "short_flow_ratio" in out.columns

    out2 = add_short_flow_factors(
        out, feature_subset=["abnormal"], short_volume_data_exists=True,
        smooth_window=5, baseline_window=20,
    )
    assert calls["finra"] == 1
    assert "short_flow_abnormal" in out2.columns


def test_short_flow_store_fetches_filing_only_when_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _make_panel(n_days=20, tickers=["AAA"])
    panel = panel.copy()
    panel["feature_date"] = pd.to_datetime(panel["date"]) - pd.Timedelta(days=1)
    calls = {"finra": 0, "filing": 0}

    def _boom_finra(*args, **kwargs):
        calls["finra"] += 1
        raise AssertionError("FINRA should not be fetched for filing-only subset")

    def _fake_filing(tickers, start_date=None, end_date=None, **kwargs):
        calls["filing"] += 1
        rows = []
        for t in tickers:
            for fd in panel.loc[panel["ticker"] == t, "feature_date"].unique():
                rows.append(
                    {
                        "date": pd.Timestamp(fd),
                        "ticker": t,
                        "last_filed": pd.Timestamp(fd) - pd.Timedelta(days=30),
                        "expected_next_filed": pd.Timestamp(fd) + pd.Timedelta(days=60),
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "data.ingestion.alternative_data.finra_short_volume.fetch_short_volume_daily",
        _boom_finra,
    )
    monkeypatch.setattr(
        "data.ingestion.alternative_data.sec_companyfacts.fetch_filing_clock_daily",
        _fake_filing,
    )
    out = add_short_flow_factors(
        panel, feature_subset=["filing_since"], filing_clock_data_exists=False
    )
    assert calls["finra"] == 0
    assert calls["filing"] == 1
    assert "last_filed" in out.columns
    assert "filing_clock_since" in out.columns


# ---------------------------------------------------------------------------
# feature_subset dispatcher smoke tests
# ---------------------------------------------------------------------------


def test_resolve_feature_subset_empty_adds_all_near_52w():
    from data.processing.feature_store import NEAR_52W_FEATURES

    panel = _make_panel(n_days=30)
    out = add_near_52w_factors(panel, window=5, feature_subset=None, normalize=False)
    for mode in NEAR_52W_FEATURES:
        assert f"near_52w_{mode}" in out.columns


def test_feature_subset_unknown_id_raises():
    panel = _make_panel(n_days=20)
    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_obv_momentum_factors(panel, feature_subset=["not_a_real_id"])


# ---------------------------------------------------------------------------
# H-009 GDELT sentiment
# ---------------------------------------------------------------------------


def _make_gdelt_panel(n_days: int = 80) -> pd.DataFrame:
    panel = _make_panel(n_days=n_days, tickers=["AAA", "BBB"])
    rng = np.random.default_rng(7)
    panel = panel.copy()
    panel["median_tone"] = rng.normal(0.0, 2.0, len(panel))
    panel["n_articles"] = rng.integers(0, 20, len(panel)).astype(float)
    return panel


def test_gdelt_tone_primitive_and_store_columns() -> None:
    from data.processing.feature_implementation.gdelt_sentiment import (
        rolling_median_tone,
        tone_momentum,
    )
    from data.processing.feature_store import add_gdelt_sentiment_factors

    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rolling_median_tone(s, window=3).iloc[2] == pytest.approx(2.0)
    mom = tone_momentum(s, short_window=2, long_window=4)
    assert mom.iloc[3] == pytest.approx(
        s.rolling(2).median().iloc[3] - s.rolling(4).median().iloc[3]
    )

    panel = _make_gdelt_panel()
    out = add_gdelt_sentiment_factors(
        panel,
        feature_subset=["tone", "attention"],
        window=5,
        sentiment_data_exists=True,
    )
    assert "gdelt_tone" in out.columns
    assert "gdelt_attention" in out.columns
    assert out["gdelt_tone"].notna().any()


def test_gdelt_abnormal_and_multi_window() -> None:
    from data.processing.feature_store import add_gdelt_sentiment_factors

    panel = _make_gdelt_panel(n_days=100)
    single = add_gdelt_sentiment_factors(
        panel,
        feature_subset=["abnormal_tone"],
        smooth_window=5,
        baseline_window=20,
        sentiment_data_exists=True,
    )
    assert "gdelt_abnormal_tone" in single.columns

    multi = add_gdelt_sentiment_factors(
        panel,
        feature_subset=["tone"],
        window=[3, 5],
        sentiment_data_exists=True,
    )
    assert "gdelt_tone_3" in multi.columns
    assert "gdelt_tone_5" in multi.columns
    assert "gdelt_tone" not in multi.columns

    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_gdelt_sentiment_factors(
            panel, feature_subset=["nope"], sentiment_data_exists=True
        )
    with pytest.raises(ValueError, match="missing columns"):
        add_gdelt_sentiment_factors(
            panel.drop(columns=["median_tone"]),
            feature_subset=["tone"],
            sentiment_data_exists=True,
        )


def test_gdelt_store_fetches_and_merges_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data.processing.feature_store import add_gdelt_sentiment_factors

    panel = _make_panel(n_days=40, tickers=["AAA", "BBB"])
    panel = panel.copy()
    panel["feature_date"] = pd.to_datetime(panel["date"]) - pd.Timedelta(days=1)

    def _fake_fetch(tickers, start_date=None, end_date=None, **kwargs):
        assert set(tickers) == {"AAA", "BBB"}
        rows = []
        for t in tickers:
            for fd in panel.loc[panel["ticker"] == t, "feature_date"].unique():
                rows.append(
                    {
                        "date": pd.Timestamp(fd),
                        "ticker": t,
                        "median_tone": 0.5,
                        "n_articles": 4,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "data.ingestion.alternative_data.sentiment.gdelt_fetcher.fetch_gdelt_sentiment_daily",
        _fake_fetch,
    )
    out = add_gdelt_sentiment_factors(
        panel, feature_subset=["tone"], window=5, sentiment_data_exists=False
    )
    assert "median_tone" in out.columns
    assert out["median_tone"].notna().all()
    assert "gdelt_tone" in out.columns
    assert out["gdelt_tone"].notna().any()


def test_gdelt_no_lookahead_prefix_stability() -> None:
    from data.processing.feature_store import add_gdelt_sentiment_factors

    panel = _make_gdelt_panel(n_days=90)
    full = add_gdelt_sentiment_factors(
        panel, feature_subset=["tone"], window=5, sentiment_data_exists=True
    )
    cutoff = panel["date"].sort_values().unique()[-20]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_gdelt_sentiment_factors(
        truncated, feature_subset=["tone"], window=5, sentiment_data_exists=True
    )
    merged = partial[["date", "ticker", "gdelt_tone"]].merge(
        full[["date", "ticker", "gdelt_tone"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    both = merged["gdelt_tone_partial"].notna() & merged["gdelt_tone_full"].notna()
    np.testing.assert_allclose(
        merged.loc[both, "gdelt_tone_partial"].to_numpy(),
        merged.loc[both, "gdelt_tone_full"].to_numpy(),
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Complementary beta / Amihud / volume / TA-Lib
# ---------------------------------------------------------------------------


def _make_spy_ohlcv(panel: pd.DataFrame) -> pd.DataFrame:
    spy = panel[panel["ticker"] == panel["ticker"].iloc[0]][
        ["date", "open", "high", "low", "close", "volume"]
    ].copy()
    spy["ticker"] = "SPY"
    return spy.reset_index(drop=True)


def test_market_corr_from_beta_r2_identity() -> None:
    from data.processing.feature_implementation.beta_features import (
        market_corr_from_beta_r2,
    )

    beta = pd.Series([1.0, -0.5, 0.0])
    r2 = pd.Series([0.25, 0.81, 0.49])
    corr = market_corr_from_beta_r2(beta, r2)
    assert corr.iloc[0] == pytest.approx(0.5)
    assert corr.iloc[1] == pytest.approx(-0.9)
    assert corr.iloc[2] == pytest.approx(0.0)


def test_beta_complements_columns_and_corr_r2() -> None:
    panel = _make_beta_panel(n_days=80)
    mkt = _make_market_returns(panel)
    spy = _make_spy_ohlcv(panel)
    out = add_beta_factors(
        panel,
        market_returns=mkt,
        market_ohlcv=spy,
        feature_subset=[
            "market_corr",
            "r_squared",
            "rel_strength",
            "beta_mkt_interact",
            "mkt_ret",
            "mkt_vol",
            "mkt_near_52w",
        ],
        windows=30,
        mkt_horizon=5,
        mkt_ret_windows=5,
        mkt_vol_windows=10,
        mkt_near_windows=20,
    )
    for col in (
        "market_corr",
        "r_squared",
        "rel_strength",
        "beta_mkt_interact",
        "mkt_ret",
        "mkt_vol",
        "mkt_near_52w",
    ):
        assert col in out.columns
        assert out[col].notna().any()

    both = out["market_corr"].notna() & out["r_squared"].notna()
    # |corr| == sqrt(r2) when r2 >= 0
    np.testing.assert_allclose(
        out.loc[both, "market_corr"].abs().to_numpy(),
        np.sqrt(out.loc[both, "r_squared"].clip(lower=0).to_numpy()),
        rtol=1e-10,
    )
    assert out["r_squared"].dropna().between(0.0, 1.0).all()
    # Market-level features identical within each date
    for col in ("mkt_ret", "mkt_vol", "mkt_near_52w"):
        nunique = out.groupby("date")[col].nunique(dropna=True)
        assert (nunique <= 1).all()


def test_beta_mkt_interact_multi_window_naming() -> None:
    panel = _make_beta_panel(n_days=80)
    mkt = _make_market_returns(panel)
    out = add_beta_factors(
        panel,
        market_returns=mkt,
        feature_subset=["beta_mkt_interact"],
        windows=[30, 40],
        mkt_horizon=[1, 5],
    )
    assert "beta_mkt_interact_30_1" in out.columns
    assert "beta_mkt_interact_40_5" in out.columns
    assert "beta_mkt_interact" not in out.columns


def test_beta_complements_require_frames() -> None:
    panel = _make_beta_panel(n_days=40)
    with pytest.raises(ValueError, match="market_returns"):
        add_beta_factors(panel, feature_subset=["market_corr"], windows=20)
    with pytest.raises(ValueError, match="market_ohlcv"):
        add_beta_factors(
            panel,
            market_returns=_make_market_returns(panel),
            feature_subset=["mkt_near_52w"],
            mkt_near_windows=20,
        )


def test_beta_complements_no_lookahead() -> None:
    panel = _make_beta_panel(n_days=90)
    mkt = _make_market_returns(panel)
    full = add_beta_factors(
        panel, market_returns=mkt, feature_subset=["rel_strength"], windows=20
    )
    cutoff = panel["date"].sort_values().unique()[-15]
    truncated = panel[panel["date"] <= cutoff].copy()
    mkt_t = mkt[mkt["date"] <= cutoff].copy()
    partial = add_beta_factors(
        truncated, market_returns=mkt_t, feature_subset=["rel_strength"], windows=20
    )
    merged = partial[["date", "ticker", "rel_strength"]].merge(
        full[["date", "ticker", "rel_strength"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    both = merged["rel_strength_partial"].notna() & merged["rel_strength_full"].notna()
    np.testing.assert_allclose(
        merged.loc[both, "rel_strength_partial"].to_numpy(),
        merged.loc[both, "rel_strength_full"].to_numpy(),
        rtol=1e-10,
    )


def test_amihud_hand_check_and_normalize() -> None:
    from data.processing.feature_implementation.size_and_valuation_features import (
        amihud_illiquidity,
    )
    from data.processing.feature_store import add_size_value_factors

    close = pd.Series([10.0, 11.0, 12.0, 13.0])
    volume = pd.Series([100.0, 100.0, 100.0, 100.0])
    # day1: |0.1|/(11*100)=1e-4; day2: |0.0909|/(12*100)≈7.576e-5
    am = amihud_illiquidity(close, volume, window=2)
    r1 = abs(11 / 10 - 1) / (11 * 100)
    r2 = abs(12 / 11 - 1) / (12 * 100)
    assert am.iloc[2] == pytest.approx((r1 + r2) / 2)

    panel = _make_panel(n_days=40)
    # Vary volume so Amihud is defined
    panel = panel.copy()
    panel["volume"] = panel["volume"] * (1.0 + 0.1 * np.arange(len(panel)) % 7)
    ranked = add_size_value_factors(
        panel, feature_subset=["amihud"], amihud_window=5, normalize=True
    )
    assert "amihud" in ranked.columns
    vals = ranked["amihud"].dropna()
    assert len(vals) > 0
    assert vals.between(0.0, 1.0).all()

    multi = add_size_value_factors(
        panel, feature_subset=["amihud"], amihud_window=[5, 10], normalize=False
    )
    assert "amihud_5" in multi.columns
    assert "amihud_10" in multi.columns


def test_abnormal_volume_store() -> None:
    from data.processing.feature_store import add_volume_factors

    panel = _make_panel(n_days=80)
    rng = np.random.default_rng(0)
    panel = panel.copy()
    panel["volume"] = rng.uniform(1e4, 5e4, len(panel))
    out = add_volume_factors(
        panel,
        feature_subset=["abnormal_volume"],
        smooth_window=5,
        baseline_window=20,
    )
    assert "abnormal_volume" in out.columns
    assert out["abnormal_volume"].notna().any()

    multi = add_volume_factors(
        panel,
        feature_subset=["abnormal_volume"],
        smooth_window=[3, 5],
        baseline_window=20,
    )
    assert "abnormal_volume_3_20" in multi.columns
    assert "abnormal_volume_5_20" in multi.columns

    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_volume_factors(panel, feature_subset=["nope"])


def test_abnormal_volume_no_lookahead() -> None:
    from data.processing.feature_store import add_volume_factors

    panel = _make_panel(n_days=90)
    rng = np.random.default_rng(1)
    panel = panel.copy()
    panel["volume"] = rng.uniform(1e4, 5e4, len(panel))
    full = add_volume_factors(
        panel, feature_subset=["abnormal_volume"], smooth_window=5, baseline_window=20
    )
    cutoff = panel["date"].sort_values().unique()[-15]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_volume_factors(
        truncated,
        feature_subset=["abnormal_volume"],
        smooth_window=5,
        baseline_window=20,
    )
    merged = partial[["date", "ticker", "abnormal_volume"]].merge(
        full[["date", "ticker", "abnormal_volume"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    both = (
        merged["abnormal_volume_partial"].notna()
        & merged["abnormal_volume_full"].notna()
    )
    np.testing.assert_allclose(
        merged.loc[both, "abnormal_volume_partial"].to_numpy(),
        merged.loc[both, "abnormal_volume_full"].to_numpy(),
        rtol=1e-10,
    )


def test_talib_factors_columns_and_multi() -> None:
    from data.processing.feature_store import add_talib_factors

    panel = _make_panel(n_days=60)
    single = add_talib_factors(
        panel, feature_subset=["rsi", "adx", "mfi", "bb_percent_b"], timeperiod=14
    )
    assert "rsi" in single.columns
    assert "adx" in single.columns
    assert "mfi" in single.columns
    assert "bb_percent_b" in single.columns
    assert single["rsi"].notna().any()
    assert single["rsi"].dropna().between(0.0, 100.0).all()

    multi = add_talib_factors(
        panel, feature_subset=["rsi"], timeperiod=[7, 14]
    )
    assert "rsi_7" in multi.columns
    assert "rsi_14" in multi.columns

    bb = add_talib_factors(
        panel,
        feature_subset=["bb_percent_b"],
        bb_timeperiod=[10, 20],
    )
    assert "bb_percent_b_10" in bb.columns
    assert "bb_percent_b_20" in bb.columns

    with pytest.raises(ValueError, match="unknown feature_subset"):
        add_talib_factors(panel, feature_subset=["macd"])
    with pytest.raises(ValueError, match="missing columns"):
        add_talib_factors(panel.drop(columns=["high"]), feature_subset=["adx"])


def test_talib_no_lookahead() -> None:
    from data.processing.feature_store import add_talib_factors

    panel = _make_panel(n_days=80)
    full = add_talib_factors(panel, feature_subset=["rsi"], timeperiod=14)
    cutoff = panel["date"].sort_values().unique()[-15]
    truncated = panel[panel["date"] <= cutoff].copy()
    partial = add_talib_factors(truncated, feature_subset=["rsi"], timeperiod=14)
    merged = partial[["date", "ticker", "rsi"]].merge(
        full[["date", "ticker", "rsi"]],
        on=["date", "ticker"],
        suffixes=("_partial", "_full"),
    )
    both = merged["rsi_partial"].notna() & merged["rsi_full"].notna()
    np.testing.assert_allclose(
        merged.loc[both, "rsi_partial"].to_numpy(),
        merged.loc[both, "rsi_full"].to_numpy(),
        rtol=1e-10,
    )
