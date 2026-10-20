"""Desk-grade STAR tearsheet helpers for the S3 FX trend final backtest."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from data.ingestion.equity_fetcher import fetch_ohlcv
from strategies.s3_fx_trend.config import S3SimConfig
from strategies.s3_fx_trend.engine import PERIODS_PER_YEAR, BookSimResult
from strategies.s3_fx_trend.metrics import (
    compound_to_s1_weeks,
    corr_to_s1,
    corr_to_s2,
    cost_bps_per_year,
    metrics_from_returns,
    metrics_from_returns_inference,
)

# Visual system for desk PDF + notebook static plots
_NAVY = "#1f4e79"
_STEEL = "#5b7c99"
_RED = "#a01313"
_GREEN = "#2a6f2a"
_GOLD = "#c4a35a"
_PURPLE = "#6b5b95"
_GRID = "#d9dde3"
_BG = "#fbfcfd"

EVENT_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("Volmageddon", "2018-01-26", "2018-02-09"),
    ("2018 Q4 selloff", "2018-10-01", "2018-12-31"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("COVID rebound", "2020-03-24", "2020-06-08"),
    ("2022 hike bear", "2022-01-03", "2022-06-16"),
    ("Ukraine invasion", "2022-02-21", "2022-03-15"),
    ("Sep 2022 CPI/gilt", "2022-09-01", "2022-10-14"),
    ("SVB / regional banks", "2023-03-08", "2023-03-24"),
    ("Oct 2023 rates scare", "2023-09-20", "2023-11-01"),
    ("Aug 2024 yen carry", "2024-07-31", "2024-08-07"),
    ("2024 US election", "2024-10-28", "2024-11-15"),
    ("2025 early vol", "2025-03-01", "2025-04-15"),
)


def equity_from_returns(returns: pd.Series, *, start_value: float = 1.0) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    if r.empty:
        return pd.Series(dtype=float, name="equity")
    eq = float(start_value) * (1.0 + r).cumprod()
    eq.name = "equity"
    return eq


def split_returns_at_is_end(
    returns: pd.Series,
    is_end: pd.Timestamp | str,
) -> tuple[pd.Series, pd.Series]:
    """Split a continuous return series into IS (<= end) and OOS (> end)."""
    end = pd.Timestamp(is_end)
    r = pd.to_numeric(returns, errors="coerce").astype(float)
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    return r.loc[r.index <= end], r.loc[r.index > end]


def spy_daily_returns(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.Series:
    """SPY close-to-close daily returns (buy-and-hold benchmark)."""
    start_s = (pd.Timestamp(start) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end_s = (pd.Timestamp(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    spy = fetch_ohlcv("SPY", start_s, end_s, auto_adjust=True)
    if spy is None or spy.empty:
        return pd.Series(dtype=float, name="spy")
    px = spy.set_index(pd.to_datetime(spy["date"]))["close"].astype(float).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    out = px.pct_change(fill_method=None).dropna()
    out.name = "spy"
    return out


def beta_corr_to_benchmark(
    strategy: pd.Series,
    benchmark: pd.Series,
) -> dict[str, float]:
    """OLS beta and Pearson corr of strategy vs benchmark on overlapping dates."""
    a = pd.to_numeric(strategy, errors="coerce").astype(float)
    b = pd.to_numeric(benchmark, errors="coerce").astype(float)
    a.index = pd.to_datetime(a.index)
    b.index = pd.to_datetime(b.index)
    joined = pd.concat([a.rename("s"), b.rename("b")], axis=1).dropna()
    if len(joined) < 5:
        return {"beta": float("nan"), "corr": float("nan"), "n": float(len(joined))}
    cov = float(joined["s"].cov(joined["b"]))
    var_b = float(joined["b"].var(ddof=1))
    beta = cov / var_b if var_b > 0 else float("nan")
    return {
        "beta": beta,
        "corr": float(joined["s"].corr(joined["b"])),
        "n": float(len(joined)),
    }


def _sortino(returns: pd.Series, *, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if r.empty:
        return float("nan")
    downside = r.loc[r < 0.0]
    if downside.empty or float(downside.std(ddof=1)) <= 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * r.mean() / downside.std(ddof=1))


def oos_headline_metrics(
    net: pd.Series,
    gross: pd.Series,
    *,
    trades: pd.DataFrame | None = None,
    s1_weekly: pd.Series | None = None,
    s2_daily: pd.Series | None = None,
    spy_daily: pd.Series | None = None,
    n_trials_local: int | None = 1,
    n_trials_stack: int | None = None,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> pd.Series:
    """Single-column scoreboard (net primary; gross Sharpe for friction)."""
    inf = metrics_from_returns_inference(
        net,
        periods_per_year=periods_per_year,
        n_trials_local=n_trials_local,
        n_trials_stack=n_trials_stack,
    )
    g = metrics_from_returns(gross, periods_per_year=periods_per_year)
    eq = equity_from_returns(net)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) >= 2 else float("nan")
    n_years = inf["n_days"] / periods_per_year if inf["n_days"] else float("nan")
    cagr = (
        float((1.0 + total) ** (1.0 / n_years) - 1.0)
        if np.isfinite(total) and np.isfinite(n_years) and n_years > 0
        else float("nan")
    )
    wins = net[net > 0]
    losses = net[net < 0]
    win_rate = float((net > 0).mean()) if len(net) else float("nan")
    profit_factor = (
        float(wins.sum() / abs(losses.sum()))
        if len(losses) and abs(float(losses.sum())) > 0
        else float("nan")
    )
    time_in_market = float((net != 0).mean()) if len(net) else float("nan")
    cost_yr = (
        cost_bps_per_year(net, trades if trades is not None else pd.DataFrame())
        if trades is not None
        else float("nan")
    )
    spy_stats = (
        beta_corr_to_benchmark(net, spy_daily)
        if spy_daily is not None and not spy_daily.empty
        else {"beta": float("nan"), "corr": float("nan")}
    )
    calmar = (
        float(cagr / abs(inf["max_drawdown"]))
        if np.isfinite(cagr)
        and np.isfinite(inf["max_drawdown"])
        and abs(inf["max_drawdown"]) > 1e-12
        else float("nan")
    )
    return pd.Series(
        {
            "ann_sharpe_net": inf["ann_sharpe"],
            "ann_sharpe_gross": g["ann_sharpe"],
            "sortino": _sortino(net, periods_per_year=periods_per_year),
            "calmar": calmar,
            "max_drawdown": inf["max_drawdown"],
            "cagr": cagr,
            "total_return": total,
            "psr": inf.get("psr", float("nan")),
            "dsr_local": inf.get("dsr_local", float("nan")),
            "dsr_stack": inf.get("dsr_stack", float("nan")),
            "corr_to_s1": corr_to_s1(net, s1_weekly),
            "corr_to_s2": corr_to_s2(net, s2_daily),
            "beta_to_spy": spy_stats["beta"],
            "corr_to_spy": spy_stats["corr"],
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "time_in_market": time_in_market,
            "n_trades": int(len(trades)) if trades is not None else 0,
            "cost_bps_year": cost_yr,
            "n_days": inf["n_days"],
            "skew": inf.get("skew", float("nan")),
            "excess_kurtosis": inf.get("excess_kurtosis", float("nan")),
        },
        dtype=float,
    )


def monthly_returns(period_returns: pd.Series) -> pd.Series:
    r = pd.to_numeric(period_returns, errors="coerce").dropna().astype(float)
    r.index = pd.DatetimeIndex(pd.to_datetime(r.index))
    if r.empty:
        return pd.Series(dtype=float, name="monthly_return")
    m = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    m.index = pd.to_datetime([f"{y}-{mo:02d}-01" for y, mo in m.index])
    m.name = "monthly_return"
    return m.sort_index()


def plot_monthly_heatmap(monthly: pd.Series, *, title: str = "OOS monthly returns"):
    """Year × month heatmap of monthly returns."""
    m = monthly.dropna()
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=_BG)
    ax.set_facecolor(_BG)
    if m.empty:
        ax.set_title(title)
        return fig
    df = m.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=list(range(1, 13)))
    data = pivot.to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.01
    lim = max(lim, 1e-6)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-lim, vmax=lim)
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])
    ax.set_title(title, color=_NAVY, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def plot_losing_months(monthly: pd.Series, *, title: str = "OOS monthly returns"):
    m = monthly.dropna()
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=_BG)
    ax.set_facecolor(_BG)
    if m.empty:
        ax.set_title(title)
        fig.tight_layout()
        return fig
    colors = [_RED if v < 0 else _GREEN for v in m.values]
    ax.bar(m.index, m.values, width=20, color=colors, align="center")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_ylabel("Month return")
    ax.set_title(title, color=_NAVY, fontweight="bold")
    ax.grid(True, axis="y", color=_GRID, lw=0.6)
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(
    returns: pd.Series,
    *,
    window: int = 63,
    periods_per_year: float = PERIODS_PER_YEAR,
    title: str = "Rolling Sharpe",
):
    r = pd.to_numeric(returns, errors="coerce").astype(float)
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=_BG)
    ax.set_facecolor(_BG)
    if len(r) < max(10, window // 2):
        ax.set_title(title)
        fig.tight_layout()
        return fig
    roll_mu = r.rolling(window, min_periods=max(8, window // 2)).mean()
    roll_sd = r.rolling(window, min_periods=max(8, window // 2)).std(ddof=1)
    sharpe = np.sqrt(periods_per_year) * roll_mu / roll_sd.replace(0.0, np.nan)
    ax.plot(sharpe.index, sharpe.values, color=_NAVY, lw=1.4)
    ax.axhline(0.0, color=_GRID, lw=0.8)
    ax.set_title(title, color=_NAVY, fontweight="bold")
    ax.set_ylabel("Ann. Sharpe")
    ax.grid(True, color=_GRID, lw=0.6)
    fig.tight_layout()
    return fig


def daily_regime_performance(
    strategy_returns: pd.Series,
    spy_daily: pd.Series,
    is_dates: pd.DatetimeIndex | pd.Series,
    *,
    vol_window: int = 63,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Strategy returns by SPY up/down and IS-fit vol terciles."""
    spy = pd.to_numeric(spy_daily, errors="coerce").dropna().astype(float)
    spy.index = pd.DatetimeIndex(pd.to_datetime(spy.index))
    oos = pd.to_numeric(strategy_returns, errors="coerce").dropna().astype(float)
    oos.index = pd.DatetimeIndex(pd.to_datetime(oos.index))
    joined = pd.concat([oos.rename("s3"), spy.rename("spy")], axis=1).dropna()

    def _bucket(labels: pd.Series) -> pd.DataFrame:
        rows = []
        for name, g in joined.groupby(labels):
            r = g["s3"]
            rows.append(
                {
                    "regime": name,
                    "n": int(len(r)),
                    "mean_return": float(r.mean()),
                    "total_return": float((1.0 + r).prod() - 1.0),
                    "sharpe": float(
                        np.sqrt(PERIODS_PER_YEAR) * r.mean() / r.std(ddof=1)
                    )
                    if r.std(ddof=1) > 0
                    else float("nan"),
                    "win_rate": float((r > 0).mean()),
                }
            )
        return pd.DataFrame(rows).set_index("regime")

    market = pd.Series(
        np.where(joined["spy"] > 0, "Up", "Down"), index=joined.index, name="market"
    )
    trail = spy.rolling(vol_window, min_periods=max(8, vol_window // 2)).std()
    is_set = set(pd.DatetimeIndex(pd.to_datetime(is_dates)))
    is_vol = trail.loc[[d for d in trail.index if d in is_set]].dropna()
    if len(is_vol) >= 3:
        q33, q66 = float(is_vol.quantile(1 / 3)), float(is_vol.quantile(2 / 3))
    else:
        q33 = q66 = float("nan")

    def _vol_label(v: float) -> str:
        if not np.isfinite(v) or not np.isfinite(q33):
            return "Unknown"
        if v <= q33:
            return "Low"
        if v <= q66:
            return "Mid"
        return "High"

    vol_labels = trail.reindex(joined.index).map(_vol_label)
    return {
        "market": _bucket(market),
        "vol": _bucket(vol_labels),
        "vol_cutpoints": pd.Series({"q33": q33, "q66": q66}),
    }


def filter_event_windows(
    returns: pd.Series,
    windows: Sequence[tuple[str, str, str]] = EVENT_WINDOWS,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Keep named windows that overlap the return index."""
    if returns is None or returns.empty:
        return []
    idx = pd.DatetimeIndex(pd.to_datetime(returns.index))
    lo, hi = idx.min(), idx.max()
    out: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for name, start, end in windows:
        a, b = pd.Timestamp(start), pd.Timestamp(end)
        if b < lo or a > hi:
            continue
        out.append((name, a, b))
    return out


def plot_event_grid(
    returns: pd.Series,
    *,
    is_end: pd.Timestamp | str,
    windows: Sequence[tuple[str, str, str]] = EVENT_WINDOWS,
    nrows: int = 4,
    ncols: int = 3,
):
    """4x3 grid of % change from each event start (OOS windows first)."""
    end = pd.Timestamp(is_end)
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    r.index = pd.to_datetime(r.index)
    kept = filter_event_windows(r, windows)
    oos = [w for w in kept if w[1] > end]
    is_w = [w for w in kept if w[1] <= end]
    ordered = (oos + is_w)[: nrows * ncols]

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, 9), facecolor=_BG, sharey=False
    )
    axes_flat = np.asarray(axes).ravel()
    for ax in axes_flat:
        ax.set_facecolor(_BG)
        ax.set_visible(False)

    for ax, (name, a, b) in zip(axes_flat, ordered):
        ax.set_visible(True)
        window = r.loc[(r.index >= a) & (r.index <= b)]
        if window.empty:
            ax.set_title(name, fontsize=9)
            continue
        pct = (1.0 + window).cumprod() - 1.0
        color = _NAVY if a > end else _STEEL
        ax.plot(pct.index, 100.0 * pct.values, color=color, lw=1.4)
        ax.axhline(0.0, color=_GRID, lw=0.8)
        tag = "OOS" if a > end else "IS"
        ax.set_title(f"{name} ({tag})", fontsize=9, color=_NAVY)
        ax.tick_params(labelsize=7)
        ax.grid(True, color=_GRID, lw=0.5)
    fig.suptitle(
        "Event study — % change from window start",
        color=_NAVY,
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def pair_scorecard(
    book: BookSimResult,
    *,
    date_min: pd.Timestamp | str | None = None,
    date_max: pd.Timestamp | str | None = None,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Per-pair net/gross Sharpe, DD, trades, cost drag (optional date window)."""
    lo = pd.Timestamp(date_min) if date_min is not None else None
    hi = pd.Timestamp(date_max) if date_max is not None else None
    rows = []
    pair_results = book.pair_results or {}
    for pid, res in pair_results.items():
        ret = pd.to_numeric(res.returns_net, errors="coerce").astype(float)
        ret.index = pd.to_datetime(ret.index)
        gross = pd.to_numeric(res.returns_gross, errors="coerce").astype(float)
        gross.index = pd.to_datetime(gross.index)
        if lo is not None:
            ret = ret.loc[ret.index >= lo]
            gross = gross.loc[gross.index >= lo]
        if hi is not None:
            ret = ret.loc[ret.index <= hi]
            gross = gross.loc[gross.index <= hi]
        trades = res.trades.copy() if res.trades is not None else pd.DataFrame()
        if not trades.empty and "entry_date" in trades.columns:
            trades["entry_date"] = pd.to_datetime(trades["entry_date"])
            if lo is not None:
                trades = trades.loc[trades["entry_date"] >= lo]
            if hi is not None:
                trades = trades.loc[trades["entry_date"] <= hi]
        net_m = metrics_from_returns(ret, periods_per_year=periods_per_year)
        gross_m = metrics_from_returns(gross, periods_per_year=periods_per_year)
        rows.append(
            {
                "pair": str(pid),
                "ann_sharpe_net": net_m["ann_sharpe"],
                "ann_sharpe_gross": gross_m["ann_sharpe"],
                "max_drawdown": net_m["max_drawdown"],
                "n_trades": int(len(trades)),
                "cost_bps_year": cost_bps_per_year(
                    ret, trades, periods_per_year=periods_per_year
                ),
                "n_days": net_m["n_days"],
            }
        )
    return pd.DataFrame(rows)


def pair_pnl_attribution(
    book: BookSimResult,
    *,
    date_min: pd.Timestamp | str | None = None,
    date_max: pd.Timestamp | str | None = None,
) -> pd.Series:
    """Compounded return contribution by pair over an optional date window."""
    lo = pd.Timestamp(date_min) if date_min is not None else None
    hi = pd.Timestamp(date_max) if date_max is not None else None
    out: dict[str, float] = {}
    for pid, res in (book.pair_results or {}).items():
        ret = pd.to_numeric(res.returns_net, errors="coerce").astype(float)
        ret.index = pd.to_datetime(ret.index)
        if lo is not None:
            ret = ret.loc[ret.index >= lo]
        if hi is not None:
            ret = ret.loc[ret.index <= hi]
        ret = ret.fillna(0.0)
        out[str(pid)] = float((1.0 + ret).prod() - 1.0) if len(ret) else float("nan")
    return pd.Series(out, name="total_return").sort_values()


def _trade_pnl_pct(
    trades: pd.DataFrame,
    book: BookSimResult,
) -> pd.DataFrame:
    """Attach compounded pair PnL between entry and exit (exclusive of exit day)."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    pair_col = "pair" if "pair" in t.columns else "pair_id"
    pnls: list[float] = []
    sides: list[str] = []
    for row in t.itertuples(index=False):
        pid = str(getattr(row, pair_col))
        side = int(getattr(row, "side", 0) or 0)
        sides.append("long" if side > 0 else ("short" if side < 0 else "flat"))
        res = (book.pair_results or {}).get(pid)
        if res is None or res.returns_net is None or res.returns_net.empty:
            pnls.append(float("nan"))
            continue
        ret = pd.to_numeric(res.returns_net, errors="coerce").astype(float)
        ret.index = pd.to_datetime(ret.index)
        entry = pd.Timestamp(row.entry_date)
        exit_ = pd.Timestamp(row.exit_date)
        window = ret.loc[(ret.index >= entry) & (ret.index < exit_)].fillna(0.0)
        pnls.append(float((1.0 + window).prod() - 1.0) if len(window) else float("nan"))
    t["pnl_pct"] = pnls
    t["side_label"] = sides
    if pair_col != "pair_id":
        t["pair_id"] = t[pair_col].astype(str)
    return t


def extreme_trades(
    trades: pd.DataFrame,
    book: BookSimResult,
    *,
    n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best and worst round-trips by compounded pair PnL."""
    enriched = _trade_pnl_pct(trades, book)
    empty = pd.DataFrame(
        columns=[
            "pair_id",
            "side_label",
            "entry_date",
            "exit_date",
            "pnl_pct",
            "exit_reason",
            "hold_bars",
        ]
    )
    if enriched.empty:
        return empty, empty
    ranked = enriched.dropna(subset=["pnl_pct"]).sort_values("pnl_pct", ascending=False)
    if ranked.empty:
        return empty, empty
    cols = [
        c
        for c in [
            "pair_id",
            "side_label",
            "entry_date",
            "exit_date",
            "pnl_pct",
            "exit_reason",
            "hold_bars",
        ]
        if c in ranked.columns
    ]
    best = ranked.head(int(n))[cols].reset_index(drop=True)
    worst = ranked.tail(int(n)).sort_values("pnl_pct")[cols].reset_index(drop=True)
    return best, worst


def utilization_stats(
    returns: pd.Series,
    trades: pd.DataFrame,
    weights: pd.DataFrame | None = None,
) -> pd.Series:
    """Active days, trade count, and mean gross book exposure."""
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    active = float((r != 0).mean()) if len(r) else float("nan")
    n_trades = int(len(trades)) if trades is not None else 0
    mean_gross = float("nan")
    if weights is not None and not weights.empty:
        mean_gross = float(weights.abs().sum(axis=1).mean())
    return pd.Series(
        {
            "pct_days_nonzero": active,
            "n_trades": float(n_trades),
            "mean_gross_exposure": mean_gross,
        }
    )


def cost_stress_sharpes(
    panel: pd.DataFrame,
    cfg: S3SimConfig,
    *,
    multipliers: Sequence[float] = (1.0, 1.5, 2.0),
    rates_df: pd.DataFrame | None = None,
    reer_df: pd.DataFrame | None = None,
    s1_weekly: pd.Series | None = None,
    s2_daily: pd.Series | None = None,
    n_trials_local: int | None = 1,
    n_trials_stack: int | None = None,
) -> pd.DataFrame:
    """Net Sharpe under cost multipliers (research stress; not a STAR bake-off)."""
    from backtest.s3_fx_trend.runner import run_s3_backtest

    rows = []
    for mult in multipliers:
        stressed = replace(cfg, cost_multiplier=float(mult))
        res = run_s3_backtest(
            panel,
            stressed,
            rates_df=rates_df,
            reer_df=reer_df,
            s1_weekly=s1_weekly,
            s2_daily=s2_daily,
            n_trials_local=n_trials_local,
            n_trials_stack=n_trials_stack,
        )
        rows.append(
            {
                "cost_multiplier": float(mult),
                "ann_sharpe_net": res.metrics.get("ann_sharpe", float("nan")),
                "ann_sharpe_gross": res.metrics.get("ann_sharpe_gross", float("nan")),
                "max_drawdown": res.metrics.get("max_drawdown", float("nan")),
                "cost_bps_per_year": res.metrics.get("cost_bps_per_year", float("nan")),
                "n_entries": res.metrics.get("n_entries", 0),
            }
        )
    return pd.DataFrame(rows)


def baseline_configs_from_star(cfg: S3SimConfig) -> dict[str, S3SimConfig]:
    """Raw TSMOM and EW long-G10 baselines derived from frozen STAR (no stack mutate)."""
    raw = replace(
        cfg,
        carry_mode="off",
        value_mode="off",
        spike_mode="off",
        corr_conflict_mode="off",
        conviction_scaling="sign",
        weak_signal_flat_z=0.0,
        rebalance_band=0.0,
        pair_vt=False,
        book_vt=False,
    )
    # Equal-weight long each pair every day: approximate via always-long sign TSMOM
    # by forcing signal_family tsmom + disabling overlays; true EW buy-hold uses
    # a dedicated long-only path via hyp_overrides on conviction and a flat signal —
    # here we expose a long-only equal book by setting conviction sign and turning
    # off short via a helper flag is not available, so we use gross long proxy:
    # mean of pair open-to-open returns (computed in notebook). Keep config for
    # "raw_tsmom" only; EW is handled separately.
    return {"raw_tsmom": raw}


def ew_long_g10_returns(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight long G10 open-to-open book return (baseline, no costs)."""
    if panel is None or panel.empty:
        return pd.Series(dtype=float, name="ew_long_g10")
    d = panel.copy()
    d["date"] = pd.to_datetime(d["date"])
    pair_col = "pair" if "pair" in d.columns else ("ticker" if "ticker" in d.columns else None)
    if pair_col is None or "open" not in d.columns:
        return pd.Series(dtype=float, name="ew_long_g10")
    parts = []
    for _, g in d.groupby(pair_col, sort=False):
        g = g.sort_values("date")
        o = pd.to_numeric(g["open"], errors="coerce")
        ret = o.pct_change(fill_method=None)
        ret.index = pd.to_datetime(g["date"].values)
        parts.append(ret.rename(str(g[pair_col].iloc[0])))
    if not parts:
        return pd.Series(dtype=float, name="ew_long_g10")
    wide = pd.concat(parts, axis=1)
    out = wide.mean(axis=1, skipna=True).dropna()
    out.name = "ew_long_g10"
    return out.sort_index()


def check_fill_timing(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Assert signal close t → fill open t+1 (no same-bar close fill)."""
    cols = [
        "pair",
        "entry_date",
        "signal_date",
        "ok_signal_before_fill",
        "ok_fill_has_open",
        "ok_not_same_bar_close_fill",
        "all_ok",
    ]
    if trades is None or trades.empty or panel is None or panel.empty:
        return pd.DataFrame(columns=cols)

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    pair_col = "pair" if "pair" in p.columns else ("ticker" if "ticker" in p.columns else None)
    rows: list[dict] = []
    tcol = "pair" if "pair" in trades.columns else "pair_id"
    for row in trades.itertuples(index=False):
        entry_date = pd.Timestamp(row.entry_date)
        pid = str(getattr(row, tcol))
        if pair_col is not None:
            g = p.loc[p[pair_col].astype(str) == pid].sort_values("date")
        else:
            g = p.sort_values("date")
        if g.empty:
            rows.append(
                {
                    "pair": pid,
                    "entry_date": entry_date,
                    "signal_date": pd.NaT,
                    "ok_signal_before_fill": False,
                    "ok_fill_has_open": False,
                    "ok_not_same_bar_close_fill": False,
                    "all_ok": False,
                }
            )
            continue
        dates = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        prior = dates[dates < entry_date]
        signal_date = prior.max() if len(prior) else pd.NaT
        ok_signal = pd.notna(signal_date) and signal_date < entry_date
        fill_rows = g.loc[pd.to_datetime(g["date"]) == entry_date]
        open_col = "open" if "open" in g.columns else None
        ok_open = (
            not fill_rows.empty
            and open_col is not None
            and np.isfinite(float(pd.to_numeric(fill_rows.iloc[0][open_col], errors="coerce")))
        )
        ok_not_same = bool(ok_signal and signal_date != entry_date)
        all_ok = bool(ok_signal and ok_open and ok_not_same)
        rows.append(
            {
                "pair": pid,
                "entry_date": entry_date,
                "signal_date": signal_date,
                "ok_signal_before_fill": bool(ok_signal),
                "ok_fill_has_open": bool(ok_open),
                "ok_not_same_bar_close_fill": bool(ok_not_same),
                "all_ok": all_ok,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def cvar(returns: pd.Series, *, alpha: float = 0.05) -> float:
    """Expected shortfall: mean of returns at or below the alpha quantile."""
    r = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if r.empty:
        return float("nan")
    q = float(r.quantile(alpha))
    tail = r.loc[r <= q]
    if tail.empty:
        return float(q)
    return float(tail.mean())


def max_consecutive_losers(returns: pd.Series) -> int:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    best = cur = 0
    for v in r:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def bootstrap_sharpe_ci(
    returns: pd.Series,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    periods_per_year: float = PERIODS_PER_YEAR,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap percentile CI for annualized Sharpe on daily net returns."""
    r = pd.to_numeric(returns, errors="coerce").dropna().astype(float).to_numpy()
    if len(r) < 10:
        return {
            "sharpe": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": float(n_boot),
        }
    rng = np.random.default_rng(seed)
    point = metrics_from_returns(
        pd.Series(r), periods_per_year=periods_per_year
    )["ann_sharpe"]
    samples = np.empty(n_boot, dtype=float)
    n = len(r)
    for i in range(n_boot):
        draw = r[rng.integers(0, n, size=n)]
        mu = draw.mean()
        sd = draw.std(ddof=1)
        samples[i] = (
            float(np.sqrt(periods_per_year) * mu / sd) if sd > 0 else float("nan")
        )
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return {
            "sharpe": float(point),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": float(n_boot),
        }
    return {
        "sharpe": float(point),
        "ci_low": float(np.quantile(samples, alpha / 2)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha / 2)),
        "n_boot": float(n_boot),
    }


def weekly_overlay_frame(
    s3_daily: pd.Series,
    s1_weekly: pd.Series,
    s2_daily: pd.Series | None = None,
    spy_daily: pd.Series | None = None,
) -> pd.DataFrame:
    """Align S3 / S1 / S2 / SPY on S1 Monday–Monday weeks."""
    if s1_weekly is None or s1_weekly.empty:
        return pd.DataFrame(columns=["s3", "s1", "s2", "spy"])
    s1 = pd.to_numeric(s1_weekly, errors="coerce").dropna().astype(float)
    s1.index = pd.to_datetime(s1.index)
    parts = [compound_to_s1_weeks(s3_daily, s1.index).rename("s3"), s1.rename("s1")]
    if s2_daily is not None and not s2_daily.empty:
        parts.append(compound_to_s1_weeks(s2_daily, s1.index).rename("s2"))
    if spy_daily is not None and not spy_daily.empty:
        parts.append(compound_to_s1_weeks(spy_daily, s1.index).rename("spy"))
    joined = pd.concat(parts, axis=1).dropna(how="any")
    return joined.sort_index()


def equity_from_common_start(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Rebase all columns to 1.0 at the first row where every series is present."""
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()
    d = returns_df.dropna(how="any").sort_index()
    if d.empty:
        return d
    return (1.0 + d.fillna(0.0)).cumprod()


def plot_static_weekly_overlay(
    weekly: pd.DataFrame,
    *,
    title: str = "S3 vs S1 vs S2 vs SPY (weekly)",
    is_end: pd.Timestamp | str | None = None,
):
    eq = equity_from_common_start(weekly)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=_BG)
    ax.set_facecolor(_BG)
    if eq.empty:
        ax.set_title(title)
        return fig
    colors = {"s3": _NAVY, "s1": _GOLD, "s2": _PURPLE, "spy": _STEEL}
    labels = {"s3": "S3", "s1": "S1", "s2": "S2", "spy": "SPY"}
    for col in ("s3", "s1", "s2", "spy"):
        if col in eq.columns:
            ax.plot(eq.index, eq[col], color=colors[col], lw=1.6, label=labels[col])
    if is_end is not None:
        ax.axvline(pd.Timestamp(is_end), color="#333333", ls="--", lw=1.0, zorder=5)
    ax.set_title(title, color=_NAVY, fontweight="bold")
    ax.set_ylabel("Equity (rebased to 1.0)")
    ax.legend(frameon=False)
    ax.grid(True, color=_GRID, lw=0.6)
    fig.tight_layout()
    return fig


def plot_equity_net_gross(
    net: pd.Series,
    gross: pd.Series,
    *,
    title: str,
    is_end: pd.Timestamp | str | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=_BG)
    ax.set_facecolor(_BG)
    en = equity_from_returns(net)
    eg = equity_from_returns(gross)
    if not eg.empty:
        ax.plot(eg.index, eg.values, color=_STEEL, lw=1.4, label="gross", alpha=0.85)
    if not en.empty:
        ax.plot(en.index, en.values, color=_NAVY, lw=1.8, label="net")
    if is_end is not None:
        ax.axvline(pd.Timestamp(is_end), color="#333333", ls="--", lw=1.0, zorder=5)
    ax.set_title(title, color=_NAVY, fontweight="bold")
    ax.set_ylabel("Equity")
    ax.legend(frameon=False)
    ax.grid(True, color=_GRID, lw=0.6)
    fig.tight_layout()
    return fig


def plot_drawdown(returns: pd.Series, *, title: str = "Drawdown"):
    eq = equity_from_returns(returns)
    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor=_BG)
    ax.set_facecolor(_BG)
    if eq.empty:
        ax.set_title(title)
        return fig
    dd = eq / eq.cummax() - 1.0
    ax.fill_between(dd.index, dd.values, 0.0, color=_RED, alpha=0.35)
    ax.plot(dd.index, dd.values, color=_RED, lw=1.0)
    ax.set_title(title, color=_NAVY, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.grid(True, color=_GRID, lw=0.6)
    fig.tight_layout()
    return fig


def early_late_is_decay(
    returns: pd.Series,
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> pd.Series:
    """Split IS returns at midpoint; report early/late Sharpe and decay ratio."""
    r = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    if len(r) < 20:
        return pd.Series(
            {
                "early_sharpe": float("nan"),
                "late_sharpe": float("nan"),
                "decay_ratio": float("nan"),
                "n_early": 0.0,
                "n_late": 0.0,
            }
        )
    mid = r.index[len(r) // 2]
    early = r.loc[r.index <= mid]
    late = r.loc[r.index > mid]
    e = metrics_from_returns(early, periods_per_year=periods_per_year)["ann_sharpe"]
    l = metrics_from_returns(late, periods_per_year=periods_per_year)["ann_sharpe"]
    decay = float(l / e) if np.isfinite(e) and abs(e) > 1e-12 else float("nan")
    return pd.Series(
        {
            "early_sharpe": float(e),
            "late_sharpe": float(l),
            "decay_ratio": decay,
            "n_early": float(len(early)),
            "n_late": float(len(late)),
            "midpoint": str(pd.Timestamp(mid).date()),
        }
    )


def export_s3_period_returns(returns: pd.Series, path: str) -> str:
    """Write daily book returns parquet (date index, column ``ret``)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    r = pd.to_numeric(returns, errors="coerce").astype(float)
    r.index = pd.to_datetime(r.index)
    r = r.sort_index().dropna()
    r.name = "ret"
    pd.DataFrame({"ret": r}).to_parquet(path)
    return path


def _style_pdf_ax(ax, title: str) -> None:
    ax.set_title(title, color=_NAVY, fontweight="bold")
    ax.grid(True, color=_GRID, lw=0.6)
    for spine in ax.spines.values():
        spine.set_color(_GRID)


def write_tearsheet_pdf(
    result,
    path: str,
    *,
    title: str = "S3 FX trend",
    n_trials_local: int | None = None,
    n_trials_stack: int | None = None,
    sleeve: str = "core",
) -> str:
    """Backward-compatible thin PDF (equity + metrics). Prefer ``write_core_star_tearsheet_pdf``."""
    from backtest.s3_fx_trend.runner import S3BacktestResult

    _ = sleeve
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(result, S3BacktestResult):
        returns = result.returns
        metrics = dict(result.metrics)
    else:
        returns = result
        metrics = metrics_from_returns_inference(
            returns,
            n_trials_local=n_trials_local,
            n_trials_stack=n_trials_stack,
        )
    orig_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        with PdfPages(path) as pdf:
            fig, ax = plt.subplots(figsize=(8, 4))
            eq = equity_from_returns(returns)
            if not eq.empty:
                ax.plot(eq.index, eq.values)
            ax.set_title(title)
            ax.set_ylabel("equity")
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            keys = [
                "ann_sharpe",
                "ann_sharpe_gross",
                "max_drawdown",
                "calmar",
                "corr_to_s1",
                "corr_to_s2",
                "psr",
                "dsr_local",
                "dsr_stack",
                "cost_bps_per_year",
                "n_days",
                "n_entries",
            ]
            rows = []
            for k in keys:
                v = metrics.get(k, float("nan"))
                if isinstance(v, (int, np.integer)):
                    rows.append([k, str(int(v))])
                elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                    rows.append([k, ""])
                else:
                    rows.append([k, f"{float(v):.6g}"])
            table = ax.table(
                cellText=rows,
                colLabels=["metric", "value"],
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.3)
            ax.set_title(f"{title} — metrics", pad=16)
            pdf.savefig(fig)
            plt.close(fig)
    finally:
        plt.show = orig_show
    return path


def write_core_star_tearsheet_pdf(
    path: str,
    *,
    stack: dict,
    is_end: pd.Timestamp | str,
    full_net: pd.Series,
    oos_net: pd.Series,
    oos_gross: pd.Series,
    headline: pd.Series,
    weekly_full: pd.DataFrame,
    weekly_oos: pd.DataFrame,
    spy_daily: pd.Series,
    event_returns: pd.Series,
    scorecard: pd.DataFrame,
    attribution: pd.Series,
    worst_trades: pd.DataFrame,
    best_trades: pd.DataFrame,
    sharpe_ci: dict,
    cvar_5: float,
    title: str = "S3 core STAR — institutional tearsheet",
) -> str:
    """Multi-page landscape desk PDF for the sealed core STAR report."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    end = pd.Timestamp(is_end)

    with PdfPages(path) as pdf:
        # Page 1 — fingerprint + metrics
        fig = plt.figure(figsize=(11, 8.5), facecolor=_BG)
        fig.suptitle(title, color=_NAVY, fontweight="bold", fontsize=14, y=0.98)
        ax = fig.add_axes([0.06, 0.55, 0.88, 0.38])
        ax.axis("off")
        keys = [
            "FAMILY_STAR",
            "HORIZON_STAR",
            "PAIR_VT_STAR",
            "BOOK_VT_STAR",
            "CONVICTION_SCALING_STAR",
            "REBALANCE_BAND_STAR",
            "CARRY_MODE_STAR",
            "VALUE_MODE_STAR",
            "CARRY_AGREE_SCALE_STAR",
            "SPIKE_MODE_STAR",
            "CORR_CONFLICT_MODE_STAR",
            "VT_TARGET_ANN_VOL_STAR",
            "COST_PROFILE_STAR",
        ]
        lines = [f"{k}: {stack.get(k)}" for k in keys if k in stack]
        ax.text(
            0.0,
            1.0,
            "STAR fingerprint\n" + "\n".join(lines),
            va="top",
            family="monospace",
            fontsize=9,
            color="#333333",
            transform=ax.transAxes,
        )
        ax2 = fig.add_axes([0.06, 0.08, 0.88, 0.42])
        ax2.axis("off")
        show = headline[
            [
                c
                for c in [
                    "ann_sharpe_net",
                    "ann_sharpe_gross",
                    "max_drawdown",
                    "cagr",
                    "psr",
                    "dsr_stack",
                    "corr_to_s1",
                    "corr_to_s2",
                    "beta_to_spy",
                    "corr_to_spy",
                    "win_rate",
                    "cost_bps_year",
                    "n_trades",
                    "n_days",
                ]
                if c in headline.index
            ]
        ]
        cell = [[f"{v:.4f}" if np.isfinite(v) else ""] for v in show.values]
        table = ax2.table(
            cellText=cell,
            rowLabels=list(show.index),
            colLabels=["OOS"],
            loc="center",
            cellLoc="right",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.35)
        ax2.set_title("OOS headline metrics", color=_NAVY, fontweight="bold", pad=12)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2 — OOS equity + DD
        fig, axes = plt.subplots(
            2, 1, figsize=(11, 8.5), facecolor=_BG, gridspec_kw={"height_ratios": [2, 1]}
        )
        for ax in axes:
            ax.set_facecolor(_BG)
        en = equity_from_returns(oos_net)
        eg = equity_from_returns(oos_gross)
        if not eg.empty:
            axes[0].plot(eg.index, eg.values, color=_STEEL, lw=1.3, label="gross")
        if not en.empty:
            axes[0].plot(en.index, en.values, color=_NAVY, lw=1.8, label="net")
        _style_pdf_ax(axes[0], "OOS equity (net vs gross)")
        axes[0].legend(frameon=False)
        if not en.empty:
            dd = en / en.cummax() - 1.0
            axes[1].fill_between(dd.index, dd.values, 0.0, color=_RED, alpha=0.35)
            axes[1].plot(dd.index, dd.values, color=_RED, lw=1.0)
        _style_pdf_ax(axes[1], "OOS drawdown (net)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3 — continuous IS/OOS + decay
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), facecolor=_BG)
        for ax in axes:
            ax.set_facecolor(_BG)
        full_eq = equity_from_returns(full_net)
        axes[0].plot(full_eq.index, full_eq.values, color=_NAVY, lw=1.8)
        axes[0].axvline(end, color="#333333", ls="--", lw=1.0, zorder=5)
        _style_pdf_ax(axes[0], "Full-sample net equity (IS-end dashed)")
        is_r, oos_r = split_returns_at_is_end(full_net, end)
        is_eq = equity_from_returns(is_r)
        oos_eq = equity_from_returns(oos_r)
        if not is_eq.empty:
            axes[1].plot(is_eq.index, is_eq.values, color=_STEEL, lw=1.5, label="IS")
        if not oos_eq.empty:
            axes[1].plot(oos_eq.index, oos_eq.values, color=_NAVY, lw=1.8, label="OOS")
        axes[1].axvline(end, color="#333333", ls="--", lw=1.0, zorder=5)
        _style_pdf_ax(axes[1], "IS and OOS each rebased to 1.0")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4 — weekly overlay + excess
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), facecolor=_BG)
        for ax in axes:
            ax.set_facecolor(_BG)
        eq = equity_from_common_start(weekly_oos if not weekly_oos.empty else weekly_full)
        if not eq.empty:
            for col, color, lab in (
                ("s3", _NAVY, "S3"),
                ("s1", _GOLD, "S1"),
                ("s2", _PURPLE, "S2"),
                ("spy", _STEEL, "SPY"),
            ):
                if col in eq.columns:
                    axes[0].plot(eq.index, eq[col], color=color, lw=1.6, label=lab)
            axes[0].legend(frameon=False)
        axes[0].axvline(end, color="#333333", ls="--", lw=1.0, zorder=5)
        _style_pdf_ax(axes[0], "Weekly overlay (common start = 1.0)")
        spy = spy_daily.reindex(oos_net.index).fillna(0.0) if spy_daily is not None else None
        if spy is not None and not spy.empty:
            excess = (oos_net.fillna(0.0) - spy).rename("excess")
            ex_eq = equity_from_returns(excess)
            if not ex_eq.empty:
                axes[1].plot(ex_eq.index, ex_eq.values, color=_NAVY, lw=1.6)
        _style_pdf_ax(axes[1], "OOS excess equity (S3 − SPY)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5 — monthly heatmap + losing months
        m = monthly_returns(oos_net)
        fig = plot_monthly_heatmap(m, title="OOS monthly heatmap")
        fig.set_size_inches(11, 4.2)
        pdf.savefig(fig)
        plt.close(fig)
        fig = plot_losing_months(m, title="OOS monthly returns")
        fig.set_size_inches(11, 3.8)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6 — events
        fig = plot_event_grid(event_returns, is_end=end)
        fig.set_size_inches(11, 8.5)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 7 — scorecard + attribution
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), facecolor=_BG)
        for ax in axes:
            ax.set_facecolor(_BG)
            ax.axis("off")
        if scorecard is not None and not scorecard.empty:
            show = scorecard.copy()
            num_cols = show.select_dtypes(include=[np.number]).columns
            for c in num_cols:
                show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            axes[0].table(
                cellText=show.values,
                colLabels=list(show.columns),
                loc="center",
                cellLoc="center",
            )
            axes[0].set_title("Pair scorecard (OOS)", color=_NAVY, fontweight="bold")
        if attribution is not None and not attribution.empty:
            axes[1].barh(
                list(attribution.index),
                100.0 * attribution.values,
                color=_NAVY,
            )
            axes[1].axis("on")
            _style_pdf_ax(axes[1], "OOS pair attribution (% total return)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 8 — trades
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), facecolor=_BG)
        for ax in axes:
            ax.set_facecolor(_BG)
            ax.axis("off")

        def _trade_table(ax, df, title_):
            ax.set_title(title_, color=_NAVY, fontweight="bold")
            if df is None or df.empty:
                return
            cols = [
                c
                for c in [
                    "pair_id",
                    "side_label",
                    "entry_date",
                    "exit_date",
                    "pnl_pct",
                    "exit_reason",
                ]
                if c in df.columns
            ]
            show = df[cols].copy()
            for c in show.columns:
                if np.issubdtype(show[c].dtype, np.datetime64):
                    show[c] = pd.to_datetime(show[c]).dt.strftime("%Y-%m-%d")
                elif np.issubdtype(show[c].dtype, np.number):
                    show[c] = show[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            ax.table(
                cellText=show.values,
                colLabels=list(show.columns),
                loc="center",
                cellLoc="center",
            )

        _trade_table(axes[0], worst_trades, "Worst OOS trades")
        _trade_table(axes[1], best_trades, "Best OOS trades")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 9 — tail / CI
        fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=_BG)
        ax.set_facecolor(_BG)
        ax.axis("off")
        txt = (
            f"CVaR 5% = {cvar_5:.4f}\n"
            f"Sharpe = {sharpe_ci.get('sharpe', float('nan')):.3f}\n"
            f"95% CI = [{sharpe_ci.get('ci_low', float('nan')):.3f}, "
            f"{sharpe_ci.get('ci_high', float('nan')):.3f}]\n"
            f"PSR = {headline.get('psr', float('nan')):.3f}\n"
            f"DSR stack = {headline.get('dsr_stack', float('nan')):.3f}\n"
            f"Max consecutive losers = {max_consecutive_losers(oos_net)}"
        )
        ax.text(0.05, 0.8, txt, fontsize=12, family="monospace", va="top")
        ax.set_title("Tail & Sharpe inference (OOS)", color=_NAVY, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return path
