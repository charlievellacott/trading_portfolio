"""H-001 research-IS diagnostics: per-pair trades, cost drag, Sharpe/DD, rolling ADF."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from data.processing.feature_implementation.cointegration import COINT_PVALUE
from strategies.s2_coint.baseline import (
    PERIODS_PER_YEAR,
    PairSimResult,
    combine_universe_returns,
    simulate_pair_baseline,
)
from strategies.s2_coint.costs import market_profile_for_pair


def metrics_from_returns(
    ret: pd.Series,
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> dict:
    """Ann. Sharpe, max drawdown, and observation count. Empty → NaNs."""
    if ret.empty:
        return {
            "ann_sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "n_days": 0,
        }
    mu = float(ret.mean())
    sd = float(ret.std(ddof=1))
    sharpe = float(np.sqrt(periods_per_year) * mu / sd) if sd > 0 else float("nan")
    equity = (1.0 + ret).cumprod()
    mdd = float((equity / equity.cummax() - 1.0).min())
    return {
        "ann_sharpe": sharpe,
        "max_drawdown": mdd,
        "n_days": int(ret.shape[0]),
    }


def cost_bps_per_year(
    ret: pd.Series,
    trades: pd.DataFrame,
    *,
    open_entry_cost_bps: float = 0.0,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> float:
    """Annualized cost drag in bps of gross book from completed (and open) fills."""
    n_days = int(ret.shape[0])
    if n_days <= 0:
        return float("nan")
    closed = 0.0
    if not trades.empty:
        closed = float(trades["entry_cost_bps"].sum() + trades["exit_cost_bps"].sum())
    years = n_days / float(periods_per_year)
    if years <= 0:
        return float("nan")
    return (closed + float(open_entry_cost_bps)) / years


def summarize_rolling_adf(
    g: pd.DataFrame,
    *,
    pvalue_threshold: float = COINT_PVALUE,
    adf_col: str = "adf_pvalue",
) -> dict:
    """IS rolling-ADF summary on a single-pair panel slice."""
    nan = float("nan")
    empty = {
        "n_adf": 0,
        "median_adf_p": nan,
        "last_adf_p": nan,
        "pct_adf_lt_threshold": nan,
        "adf_threshold": float(pvalue_threshold),
    }
    if g.empty or adf_col not in g.columns:
        return empty
    s = pd.to_numeric(g[adf_col], errors="coerce")
    finite = s[np.isfinite(s.to_numpy(dtype=float))]
    n = int(finite.shape[0])
    if n == 0:
        return empty
    return {
        "n_adf": n,
        "median_adf_p": float(finite.median()),
        "last_adf_p": float(finite.iloc[-1]),
        "pct_adf_lt_threshold": float((finite < pvalue_threshold).mean()),
        "adf_threshold": float(pvalue_threshold),
    }


def pair_diagnostics(
    g: pd.DataFrame,
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
    adf_pvalue_threshold: float = COINT_PVALUE,
    **sim_kwargs,
) -> dict:
    """One locked pair on the caller-supplied panel (research IS only)."""
    result: PairSimResult = simulate_pair_baseline(g, **sim_kwargs)
    m = metrics_from_returns(result.returns, periods_per_year=periods_per_year)
    trades = result.trades
    median_hold = (
        float(trades["hold_bars"].median()) if not trades.empty else float("nan")
    )
    cost_yr = cost_bps_per_year(
        result.returns,
        trades,
        open_entry_cost_bps=result.open_entry_cost_bps,
        periods_per_year=periods_per_year,
    )
    adf = summarize_rolling_adf(g, pvalue_threshold=adf_pvalue_threshold)
    ty = str(g["ticker_y"].iloc[0]) if not g.empty else ""
    tx = str(g["ticker_x"].iloc[0]) if not g.empty else ""
    pid = result.pair_id or (str(g["pair_id"].iloc[0]) if not g.empty else "")
    profile = ""
    if ty and tx and pid:
        profile = market_profile_for_pair(pid, ty, tx)
    return {
        "pair_id": pid,
        "ticker_y": ty,
        "ticker_x": tx,
        "market_profile": profile,
        "n_entries": int(result.n_entries),
        "n_round_trips": int(len(trades)),
        "n_open_at_end": int(result.n_open_at_end),
        "median_hold_bars": median_hold,
        "cost_bps_year": cost_yr,
        "ann_sharpe": m["ann_sharpe"],
        "max_drawdown": m["max_drawdown"],
        "n_days": m["n_days"],
        **adf,
    }


def diagnose_locked_panel(
    panel: pd.DataFrame,
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
    adf_pvalue_threshold: float = COINT_PVALUE,
    **sim_kwargs,
) -> pd.DataFrame:
    """Per-pair IS diagnostic table. Empty panel → empty frame with schema."""
    cols = [
        "pair_id",
        "ticker_y",
        "ticker_x",
        "market_profile",
        "n_entries",
        "n_round_trips",
        "n_open_at_end",
        "median_hold_bars",
        "cost_bps_year",
        "ann_sharpe",
        "max_drawdown",
        "n_days",
        "n_adf",
        "median_adf_p",
        "last_adf_p",
        "pct_adf_lt_threshold",
        "adf_threshold",
    ]
    if panel.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, g in panel.groupby("pair_id", sort=False):
        rows.append(
            pair_diagnostics(
                g,
                periods_per_year=periods_per_year,
                adf_pvalue_threshold=adf_pvalue_threshold,
                **sim_kwargs,
            )
        )
    return pd.DataFrame(rows, columns=cols)


def universe_is_metrics(
    panel: pd.DataFrame,
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
    s1_weekly: pd.Series | None = None,
    **sim_kwargs,
) -> dict:
    """Equal-weight book metrics on the supplied IS panel."""
    ret = combine_universe_returns(panel, **sim_kwargs)
    m = metrics_from_returns(ret, periods_per_year=periods_per_year)
    m["corr_to_s1"] = corr_to_s1(ret, s1_weekly)
    return m


def load_s1_period_returns(path: str) -> pd.Series:
    """Weekly S1 net returns (Monday entry index, column ``ret``). Missing → empty."""
    if not path or not os.path.isfile(path):
        return pd.Series(dtype=float, name="s1")
    df = pd.read_parquet(path)
    if "ret" in df.columns:
        s = df["ret"]
    else:
        s = df.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index)
    out = s.dropna().sort_index().astype(float)
    out.name = "s1"
    return out


def compound_to_s1_weeks(s2_daily: pd.Series, s1_index: pd.DatetimeIndex) -> pd.Series:
    """Compound S2 daily net returns over each S1 hold ``[T, T_next)`` (Mon–Mon)."""
    if s2_daily is None or s2_daily.empty or len(s1_index) == 0:
        return pd.Series(dtype=float, name="s2_week")
    s2 = pd.to_numeric(s2_daily, errors="coerce").dropna().astype(float)
    s2.index = pd.to_datetime(s2.index)
    s2 = s2.sort_index()
    weeks = pd.DatetimeIndex(pd.to_datetime(s1_index)).sort_values().unique()
    out: dict[pd.Timestamp, float] = {}
    for i, t in enumerate(weeks):
        t0 = pd.Timestamp(t)
        t1 = (
            pd.Timestamp(weeks[i + 1])
            if i + 1 < len(weeks)
            else t0 + pd.Timedelta(days=7)
        )
        window = s2.loc[(s2.index >= t0) & (s2.index < t1)]
        if window.empty:
            continue
        out[t0] = float((1.0 + window).prod() - 1.0)
    return pd.Series(out, dtype=float, name="s2_week")


def corr_to_s1(s2_daily: pd.Series, s1_weekly: pd.Series | None) -> float:
    """Pearson corr of S2 Mon–Mon compounds vs S1 weekly returns. Sparse → NaN."""
    if (
        s2_daily is None
        or s1_weekly is None
        or s2_daily.empty
        or s1_weekly.empty
    ):
        return float("nan")
    s1 = pd.to_numeric(s1_weekly, errors="coerce").dropna().astype(float)
    s1.index = pd.to_datetime(s1.index)
    s1 = s1.sort_index()
    weekly = compound_to_s1_weeks(s2_daily, s1.index)
    joined = pd.concat([weekly.rename("s2"), s1.rename("s1")], axis=1).dropna()
    if len(joined) < 3:
        return float("nan")
    return float(joined["s2"].corr(joined["s1"]))
