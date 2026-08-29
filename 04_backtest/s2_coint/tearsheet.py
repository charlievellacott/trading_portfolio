"""Desk-grade STAR tearsheet helpers for the S2 cointegration final backtest."""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from backtest.s2_coint.diagnosis import (
    gross_returns_from_net,
)
from data.ingestion.equity_fetcher import fetch_ohlcv
from data.processing.feature_implementation.cointegration import COINT_PVALUE
from performance.metrics import calmar_ratio, sortino_ratio
from strategies.s2_coint.baseline import PERIODS_PER_YEAR
from strategies.s2_coint.metrics import (
    compound_to_s1_weeks,
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


def fit_mean_abs_score(
    panel: pd.DataFrame,
    *,
    score_column: str = "z",
    date_mask: pd.Series | None = None,
) -> float:
    """IS mean of ``|score|`` for score sizing; returns 1.0 when empty/degenerate."""
    if panel is None or panel.empty or score_column not in panel.columns:
        return 1.0
    d = panel
    if date_mask is not None:
        d = panel.loc[date_mask.astype(bool)]
    s = pd.to_numeric(d[score_column], errors="coerce").astype(float).abs()
    m = float(s[np.isfinite(s)].mean()) if len(s) else float("nan")
    if not np.isfinite(m) or m <= 0:
        return 1.0
    return m


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


def oos_headline_metrics(
    net: pd.Series,
    gross: pd.Series,
    *,
    trades: pd.DataFrame | None = None,
    s1_weekly: pd.Series | None = None,
    spy_daily: pd.Series | None = None,
    n_trials_local: int | None = 1,
    n_trials_stack: int | None = None,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> pd.Series:
    """Single-column OOS scoreboard (net primary; gross Sharpe for friction)."""
    from strategies.s2_coint.metrics import corr_to_s1

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
    return pd.Series(
        {
            "ann_sharpe_net": inf["ann_sharpe"],
            "ann_sharpe_gross": g["ann_sharpe"],
            "sortino": sortino_ratio(net),
            "calmar": calmar_ratio(net),
            "max_drawdown": inf["max_drawdown"],
            "cagr": cagr,
            "total_return": total,
            "psr": inf.get("psr", float("nan")),
            "dsr_local": inf.get("dsr_local", float("nan")),
            "dsr_stack": inf.get("dsr_stack", float("nan")),
            "corr_to_s1": corr_to_s1(net, s1_weekly),
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


def daily_regime_performance(
    oos_returns: pd.Series,
    spy_daily: pd.Series,
    is_dates: pd.DatetimeIndex | pd.Series,
    *,
    vol_window: int = 63,
) -> dict[str, pd.DataFrame | pd.Series]:
    """OOS daily returns by SPY up/down and IS-fit vol terciles."""
    spy = pd.to_numeric(spy_daily, errors="coerce").dropna().astype(float)
    spy.index = pd.DatetimeIndex(pd.to_datetime(spy.index))
    oos = pd.to_numeric(oos_returns, errors="coerce").dropna().astype(float)
    oos.index = pd.DatetimeIndex(pd.to_datetime(oos.index))
    joined = pd.concat([oos.rename("s2"), spy.rename("spy")], axis=1).dropna()

    def _bucket(labels: pd.Series) -> pd.DataFrame:
        rows = []
        for name, g in joined.groupby(labels):
            r = g["s2"]
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


def reversion_health_table(
    panel: pd.DataFrame,
    *,
    is_end: pd.Timestamp | str,
    adf_threshold: float = COINT_PVALUE,
) -> pd.DataFrame:
    """Per-pair IS vs OOS ADF / half-life / beta-drift health."""
    end = pd.Timestamp(is_end)
    rows = []
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    for pid, g in p.groupby("pair_id", sort=False):
        g = g.sort_values("date")
        for label, mask in (
            ("IS", g["date"] <= end),
            ("OOS", g["date"] > end),
        ):
            sub = g.loc[mask]
            adf = pd.to_numeric(sub.get("adf_pvalue"), errors="coerce")
            hl = pd.to_numeric(sub.get("half_life"), errors="coerce")
            beta = pd.to_numeric(sub.get("beta"), errors="coerce")
            adf_fin = adf[np.isfinite(adf)]
            hl_fin = hl[np.isfinite(hl)]
            beta_fin = beta[np.isfinite(beta)]
            rows.append(
                {
                    "pair_id": str(pid),
                    "window": label,
                    "pct_adf_lt_threshold": float((adf_fin < adf_threshold).mean() * 100)
                    if len(adf_fin)
                    else float("nan"),
                    "median_half_life": float(hl_fin.median()) if len(hl_fin) else float("nan"),
                    "beta_std": float(beta_fin.std(ddof=1)) if len(beta_fin) > 1 else float("nan"),
                    "n_days": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def pair_scorecard(
    book_pair_results: dict,
    panel: pd.DataFrame,
    *,
    is_end: pd.Timestamp | str,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """OOS per-pair net/gross Sharpe, trades, cost drag, ADF health."""
    end = pd.Timestamp(is_end)
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    rows = []
    for pid, res in book_pair_results.items():
        ret = pd.to_numeric(res.returns, errors="coerce").astype(float)
        ret.index = pd.to_datetime(ret.index)
        oos = ret.loc[ret.index > end]
        trades = res.trades.copy() if res.trades is not None else pd.DataFrame()
        if not trades.empty:
            trades["entry_date"] = pd.to_datetime(trades["entry_date"])
            trades = trades.loc[trades["entry_date"] > end]
        gross = gross_returns_from_net(oos, trades, open_entry_cost_bps=res.open_entry_cost_bps)
        net_m = metrics_from_returns(oos, periods_per_year=periods_per_year)
        gross_m = metrics_from_returns(gross, periods_per_year=periods_per_year)
        g = p.loc[(p["pair_id"].astype(str) == str(pid)) & (p["date"] > end)]
        adf = pd.to_numeric(g.get("adf_pvalue"), errors="coerce")
        adf_fin = adf[np.isfinite(adf)]
        rows.append(
            {
                "pair_id": str(pid),
                "ann_sharpe_net": net_m["ann_sharpe"],
                "ann_sharpe_gross": gross_m["ann_sharpe"],
                "max_drawdown": net_m["max_drawdown"],
                "n_trades": int(len(trades)),
                "cost_bps_year": cost_bps_per_year(
                    oos, trades, open_entry_cost_bps=0.0, periods_per_year=periods_per_year
                ),
                "pct_adf_lt_05": float((adf_fin < 0.05).mean() * 100)
                if len(adf_fin)
                else float("nan"),
                "n_days": net_m["n_days"],
            }
        )
    return pd.DataFrame(rows)


def pair_pnl_attribution(
    book_pair_results: dict,
    *,
    is_end: pd.Timestamp | str,
) -> pd.Series:
    """OOS total compounded return contribution by pair."""
    end = pd.Timestamp(is_end)
    out = {}
    for pid, res in book_pair_results.items():
        ret = pd.to_numeric(res.returns, errors="coerce").astype(float)
        ret.index = pd.to_datetime(ret.index)
        oos = ret.loc[ret.index > end].fillna(0.0)
        out[str(pid)] = float((1.0 + oos).prod() - 1.0) if len(oos) else float("nan")
    return pd.Series(out, name="oos_total_return").sort_values()


def cvar(
    returns: pd.Series,
    *,
    alpha: float = 0.05,
) -> float:
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


def _hedge_weights(side: int, beta: float, use_hedge: bool = True) -> tuple[float, float]:
    y_w = float(side)
    x_w = -float(side) * float(beta) if use_hedge else -float(side)
    gross = abs(y_w) + abs(x_w)
    if gross > 0:
        y_w /= gross
        x_w /= gross
    return y_w, x_w


def gap_vs_path_decomposition(
    panel: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    is_end: pd.Timestamp | str | None = None,
    use_hedge_ratio_sizing: bool = True,
) -> pd.DataFrame:
    """Split open-to-open PnL into session / weekday-overnight / weekend buckets.

    For each hold day of each trade:
    - session: open → close
    - weekday_overnight: close → next open with no Saturday/Sunday in between
    - weekend: close → next open that spans a weekend (or holiday gap with Sat/Sun)
    """
    cols = [
        "bucket",
        "sum_pnl",
        "pct_of_pnl",
        "pct_of_variance",
        "mean_return",
        "n_bars",
    ]
    empty = pd.DataFrame(columns=cols)
    if panel is None or panel.empty or trades is None or trades.empty:
        return empty

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    end = pd.Timestamp(is_end) if is_end is not None else None
    buckets = {
        "session": [],
        "weekday_overnight": [],
        "weekend": [],
    }

    for row in trades.itertuples(index=False):
        pid = str(row.pair_id)
        side = int(row.side)
        entry = pd.Timestamp(row.entry_date)
        exit_d = pd.Timestamp(row.exit_date)
        if end is not None and entry <= end:
            continue
        g = p.loc[p["pair_id"].astype(str) == pid].sort_values("date").reset_index(drop=True)
        if g.empty:
            continue
        dates = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        # Hold PnL accrues on fill dates from entry through the bar before exit.
        for i in range(len(g) - 1):
            d0 = dates[i]
            d1 = dates[i + 1]
            if d0 < entry or d0 >= exit_d:
                continue
            beta = float(g["beta"].iloc[i]) if "beta" in g.columns else 1.0
            if not np.isfinite(beta):
                continue
            y_w, x_w = _hedge_weights(side, beta, use_hedge_ratio_sizing)
            oy0 = float(g["open_y"].iloc[i])
            ox0 = float(g["open_x"].iloc[i])
            cy0 = float(g["close_y"].iloc[i])
            cx0 = float(g["close_x"].iloc[i])
            oy1 = float(g["open_y"].iloc[i + 1])
            ox1 = float(g["open_x"].iloc[i + 1])
            if not all(np.isfinite(v) and v > 0 for v in (oy0, ox0, cy0, cx0, oy1, ox1)):
                continue
            session = y_w * (cy0 / oy0 - 1.0) + x_w * (cx0 / ox0 - 1.0)
            overnight = y_w * (oy1 / cy0 - 1.0) + x_w * (ox1 / cx0 - 1.0)
            spans_weekend = any(
                (d0 + pd.Timedelta(days=k)).dayofweek >= 5
                for k in range(1, max(int((d1 - d0).days), 1) + 1)
            )
            buckets["session"].append(session)
            if spans_weekend:
                buckets["weekend"].append(overnight)
            else:
                buckets["weekday_overnight"].append(overnight)

    all_vals = (
        buckets["session"] + buckets["weekday_overnight"] + buckets["weekend"]
    )
    total_pnl = float(np.sum(all_vals)) if all_vals else 0.0
    total_var = float(np.var(all_vals, ddof=1)) if len(all_vals) > 1 else float("nan")
    rows = []
    for name, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        sum_pnl = float(arr.sum()) if len(arr) else 0.0
        var = float(arr.var(ddof=1)) if len(arr) > 1 else float("nan")
        rows.append(
            {
                "bucket": name,
                "sum_pnl": sum_pnl,
                "pct_of_pnl": (100.0 * sum_pnl / total_pnl)
                if abs(total_pnl) > 1e-12
                else float("nan"),
                "pct_of_variance": (100.0 * var / total_var)
                if np.isfinite(var) and np.isfinite(total_var) and total_var > 0
                else float("nan"),
                "mean_return": float(arr.mean()) if len(arr) else float("nan"),
                "n_bars": int(len(arr)),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def plot_gap_bucket_means(gap_table: pd.DataFrame, *, title: str = "Gap vs path"):
    """Bar chart of mean return per bar for session / weekday overnight / weekend."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4), facecolor=_BG)
    ax.set_facecolor(_BG)
    if gap_table is None or gap_table.empty:
        ax.set_title(title)
        return fig
    order = ["session", "weekday_overnight", "weekend"]
    t = gap_table.set_index("bucket").reindex(order)
    colors = [_NAVY, _STEEL, _RED]
    vals = [10000.0 * float(v) if np.isfinite(v) else 0.0 for v in t["mean_return"]]
    ax.bar(order, vals, color=colors)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_ylabel("Mean return (bps)")
    ax.set_title(title, color=_NAVY, fontweight="bold")
    ax.grid(True, axis="y", color=_GRID, lw=0.5)
    fig.tight_layout()
    return fig


def capacity_table(
    tickers: Sequence[str],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    book_notional: float = 1_000_000.0,
    n_slots: int = 6,
    adv_window: int = 20,
) -> pd.DataFrame:
    """20d dollar ADV and % ADV at equal slot notional for each leg."""
    start_s = (pd.Timestamp(start) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    slot = float(book_notional) / max(int(n_slots), 1)
    rows = []
    for tkr in tickers:
        raw = fetch_ohlcv(str(tkr), start_s, end_s, auto_adjust=False)
        if raw is None or raw.empty:
            rows.append(
                {
                    "ticker": str(tkr),
                    "adv_usd": float("nan"),
                    "slot_notional": slot,
                    "pct_adv": float("nan"),
                    "thin_flag": True,
                }
            )
            continue
        d = raw.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date")
        d["dollar_vol"] = d["close"].astype(float) * d["volume"].astype(float)
        adv = float(d["dollar_vol"].tail(adv_window).mean())
        pct = 100.0 * slot / adv if np.isfinite(adv) and adv > 0 else float("nan")
        rows.append(
            {
                "ticker": str(tkr),
                "adv_usd": adv,
                "slot_notional": slot,
                "pct_adv": pct,
                "thin_flag": bool(
                    str(tkr).endswith((".A", ".B")) or str(tkr) in {"NWSA", "NWS"}
                ),
            }
        )
    return pd.DataFrame(rows)


def dividend_split_audit(
    panel: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    split_jump: float = 0.15,
) -> pd.DataFrame:
    """Flag trades overlapping large raw-close jumps (proxy for splits/dividends).

    Universe D panels are unadjusted; compare raw close vs adjusted close when both
    can be fetched, and also flag |raw close pct change| > ``split_jump``.
    """
    cols = [
        "pair_id",
        "entry_date",
        "exit_date",
        "pnl_pct",
        "n_jump_days",
        "max_abs_jump",
        "tickers_flagged",
    ]
    if trades is None or trades.empty or panel is None or panel.empty:
        return pd.DataFrame(columns=cols)

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    tickers = sorted(
        set(p["ticker_y"].astype(str)).union(set(p["ticker_x"].astype(str)))
    )
    start = p["date"].min() - pd.Timedelta(days=5)
    end = p["date"].max() + pd.Timedelta(days=5)
    jump_by_ticker: dict[str, pd.Series] = {}
    for tkr in tickers:
        raw = fetch_ohlcv(
            tkr,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
        adj = fetch_ohlcv(
            tkr,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            continue
        raw_px = raw.set_index(pd.to_datetime(raw["date"]))["close"].astype(float)
        raw_px = raw_px[~raw_px.index.duplicated(keep="last")].sort_index()
        jump = raw_px.pct_change(fill_method=None).abs()
        if adj is not None and not adj.empty:
            adj_px = adj.set_index(pd.to_datetime(adj["date"]))["close"].astype(float)
            adj_px = adj_px[~adj_px.index.duplicated(keep="last")].sort_index()
            # Days where raw moved a lot more than adj (corp action on unadjusted).
            joined = pd.concat(
                [raw_px.rename("raw"), adj_px.rename("adj")], axis=1
            ).dropna()
            if len(joined) >= 2:
                gap = (
                    joined["raw"].pct_change(fill_method=None)
                    - joined["adj"].pct_change(fill_method=None)
                ).abs()
                jump = pd.concat([jump, gap], axis=1).max(axis=1)
        jump_by_ticker[tkr] = jump

    rows = []
    for row in trades.itertuples(index=False):
        pid = str(row.pair_id)
        entry = pd.Timestamp(row.entry_date)
        exit_d = pd.Timestamp(row.exit_date)
        legs = pid.split("|")
        flagged = []
        jumps = []
        for tkr in legs:
            series = jump_by_ticker.get(tkr)
            if series is None:
                continue
            window = series.loc[(series.index >= entry) & (series.index <= exit_d)]
            big = window.loc[window >= split_jump]
            if not big.empty:
                flagged.append(tkr)
                jumps.extend(list(big.values))
        rows.append(
            {
                "pair_id": pid,
                "entry_date": entry,
                "exit_date": exit_d,
                "pnl_pct": float(getattr(row, "pnl_pct", float("nan"))),
                "n_jump_days": int(len(jumps)),
                "max_abs_jump": float(np.max(jumps)) if jumps else 0.0,
                "tickers_flagged": ",".join(flagged),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def weekly_overlay_frame(
    s2_daily: pd.Series,
    s1_weekly: pd.Series,
    spy_daily: pd.Series,
) -> pd.DataFrame:
    """Align S2 / S1 / SPY on S1 Monday–Monday weeks; equity curves from common start."""
    if s1_weekly is None or s1_weekly.empty:
        return pd.DataFrame(columns=["s2", "s1", "spy"])
    s1 = pd.to_numeric(s1_weekly, errors="coerce").dropna().astype(float)
    s1.index = pd.to_datetime(s1.index)
    s2w = compound_to_s1_weeks(s2_daily, s1.index)
    spy_w = compound_to_s1_weeks(spy_daily, s1.index)
    joined = pd.concat(
        [s2w.rename("s2"), s1.rename("s1"), spy_w.rename("spy")], axis=1
    ).dropna(how="any")
    return joined.sort_index()


def equity_from_common_start(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Rebase all columns to 1.0 at the first row where every series is present."""
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()
    d = returns_df.dropna(how="any").sort_index()
    if d.empty:
        return d
    return (1.0 + d.fillna(0.0)).cumprod()


def plotly_weekly_overlay(
    full_weekly: pd.DataFrame,
    oos_weekly: pd.DataFrame,
    *,
    title: str = "S2 vs S1 vs SPY (weekly)",
):
    """Interactive equity overlay with OOS / IS+OOS toggle; all start at 1.0."""
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = {"s2": _NAVY, "s1": _GOLD, "spy": _STEEL}
    labels = {"s2": "S2", "s1": "S1", "spy": "SPY"}

    def _add(df: pd.DataFrame, visible: bool):
        eq = equity_from_common_start(df)
        if eq.empty:
            return
        for col in ("s2", "s1", "spy"):
            if col not in eq.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=eq.index,
                    y=eq[col],
                    name=labels[col],
                    line=dict(color=colors[col], width=2),
                    visible=visible,
                    legendgroup=col,
                )
            )

    _add(oos_weekly, visible=True)
    _add(full_weekly, visible=False)
    n = 3
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        hovermode="x unified",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.18,
                buttons=[
                    dict(
                        label="OOS",
                        method="update",
                        args=[
                            {"visible": [True] * n + [False] * n},
                            {"title": f"{title} — OOS"},
                        ],
                    ),
                    dict(
                        label="IS+OOS",
                        method="update",
                        args=[
                            {"visible": [False] * n + [True] * n},
                            {"title": f"{title} — IS+OOS"},
                        ],
                    ),
                ],
            )
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Equity (rebased to 1.0)",
    )
    return fig


def plot_static_weekly_overlay(
    weekly: pd.DataFrame,
    *,
    title: str = "S2 vs S1 vs SPY (weekly)",
    is_end: pd.Timestamp | str | None = None,
):
    eq = equity_from_common_start(weekly)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=_BG)
    ax.set_facecolor(_BG)
    if eq.empty:
        ax.set_title(title)
        return fig
    ax.plot(eq.index, eq["s2"], color=_NAVY, lw=1.8, label="S2")
    if "s1" in eq.columns:
        ax.plot(eq.index, eq["s1"], color=_GOLD, lw=1.5, label="S1")
    if "spy" in eq.columns:
        ax.plot(eq.index, eq["spy"], color=_STEEL, lw=1.5, label="SPY")
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


def utilization_stats(
    returns: pd.Series,
    trades: pd.DataFrame,
    *,
    n_slots: int = 6,
) -> pd.Series:
    """Rough utilization under freeze + never_allow (active days / calendar)."""
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    active = float((r != 0).mean()) if len(r) else float("nan")
    n_trades = int(len(trades)) if trades is not None else 0
    return pd.Series(
        {
            "pct_days_nonzero": active,
            "n_trades": n_trades,
            "n_slots": float(n_slots),
            "implied_slot_fill": active,  # book already averages slot weights
        }
    )


def cost_stress_oos_sharpes(
    panel: pd.DataFrame,
    cfg,
    *,
    mean_abs_score: float,
    s1_weekly: pd.Series | None = None,
    n_trials_local: int = 1,
    n_trials_stack: int | None = None,
) -> pd.DataFrame:
    """OOS net Sharpe under cost stress (not a STAR bake-off)."""
    from backtest.s2_coint.runner import run_s2_backtest
    from strategies.s2_coint.costs import COSTS

    profile_key = cfg.cost_profile or "US_ALPACA_D_REALISTIC"
    if profile_key not in COSTS:
        profile_key = "US_ALPACA_D_REALISTIC"
    base = dict(COSTS[profile_key])
    scenarios = [
        (
            "baseline",
            base.get("slippage_bps_per_leg", 3.2),
            base.get("alt_slippage_bps_per_leg", 8.0),
            base.get("borrow_bps_annual", 100.0),
        ),
        (
            "2x_slippage",
            2.0 * float(base.get("slippage_bps_per_leg", 3.2)),
            2.0 * float(base.get("alt_slippage_bps_per_leg", 8.0)),
            float(base.get("borrow_bps_annual", 100.0)),
        ),
        (
            "borrow_0",
            float(base.get("slippage_bps_per_leg", 3.2)),
            float(base.get("alt_slippage_bps_per_leg", 8.0)),
            0.0,
        ),
        (
            "borrow_100",
            float(base.get("slippage_bps_per_leg", 3.2)),
            float(base.get("alt_slippage_bps_per_leg", 8.0)),
            100.0,
        ),
        (
            "borrow_200",
            float(base.get("slippage_bps_per_leg", 3.2)),
            float(base.get("alt_slippage_bps_per_leg", 8.0)),
            200.0,
        ),
    ]
    rows = []
    try:
        for name, slip, alt_slip, borrow in scenarios:
            COSTS[profile_key] = {
                **base,
                "slippage_bps_per_leg": float(slip),
                "alt_slippage_bps_per_leg": float(alt_slip),
                "borrow_bps_annual": float(borrow),
            }
            res = run_s2_backtest(
                panel,
                cfg,
                s1_weekly=s1_weekly,
                mean_abs_score=mean_abs_score,
                n_trials_local=n_trials_local,
                n_trials_stack=n_trials_stack,
            )
            rows.append(
                {
                    "scenario": name,
                    "slippage_bps": float(slip),
                    "alt_slippage_bps": float(alt_slip),
                    "borrow_bps_annual": float(borrow),
                    "ann_sharpe_net": res.metrics["ann_sharpe"],
                    "max_drawdown": res.metrics["max_drawdown"],
                }
            )
    finally:
        COSTS[profile_key] = base
    return pd.DataFrame(rows)


def export_s2_period_returns(
    returns: pd.Series,
    path: str,
) -> str:
    """Write daily net book returns parquet (date index, column ``ret``)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    s = pd.to_numeric(returns, errors="coerce").astype(float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.name = "ret"
    df = s.to_frame()
    df.index.name = "date"
    df.to_parquet(path)
    return path


def _style_pdf_ax(ax, title: str):
    ax.set_facecolor(_BG)
    ax.set_title(title, color=_NAVY, fontweight="bold", fontsize=11)
    ax.grid(True, color=_GRID, lw=0.5)
    for spine in ax.spines.values():
        spine.set_color(_GRID)


def write_star_tearsheet_pdf(
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
    health: pd.DataFrame,
    attribution: pd.Series,
    worst_trades: pd.DataFrame,
    best_trades: pd.DataFrame,
    capacity: pd.DataFrame,
    gap_table: pd.DataFrame,
    sharpe_ci: dict,
    cvar_5: float,
    title: str = "S2 STAR — institutional tearsheet",
) -> str:
    """Multi-page landscape desk PDF (coordinated panels, not one plot per page)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    end = pd.Timestamp(is_end)

    with PdfPages(path) as pdf:
        # Page 1 — fingerprint + metrics
        fig = plt.figure(figsize=(11, 8.5), facecolor=_BG)
        fig.suptitle(title, color=_NAVY, fontweight="bold", fontsize=14, y=0.98)
        ax = fig.add_axes([0.06, 0.55, 0.88, 0.38])
        ax.axis("off")
        keys = [
            "UNIVERSE_STAR",
            "BAR_STAR",
            "PAIRS_STAR",
            "BREAK_STAR",
            "ENTRY_Z_STAR",
            "Z_WINDOW_STAR",
            "EXIT_STAR",
            "OVERLAP_STAR",
            "SIZE_STAR",
            "VOL_STAR",
            "ENTRY_STAR",
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
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), facecolor=_BG, gridspec_kw={"height_ratios": [2, 1]})
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
            axes[0].plot(eq.index, eq["s2"], color=_NAVY, lw=1.8, label="S2")
            if "s1" in eq.columns:
                axes[0].plot(eq.index, eq["s1"], color=_GOLD, lw=1.5, label="S1")
            if "spy" in eq.columns:
                axes[0].plot(eq.index, eq["spy"], color=_STEEL, lw=1.5, label="SPY")
            axes[0].legend(frameon=False)
        axes[0].axvline(end, color="#333333", ls="--", lw=1.0, zorder=5)
        _style_pdf_ax(axes[0], "Weekly overlay (common start = 1.0)")
        # Excess S2 - SPY daily OOS
        spy = spy_daily.reindex(oos_net.index).fillna(0.0)
        excess = (oos_net.fillna(0.0) - spy).rename("excess")
        ex_eq = equity_from_returns(excess)
        if not ex_eq.empty:
            axes[1].plot(ex_eq.index, ex_eq.values, color=_NAVY, lw=1.6)
        _style_pdf_ax(axes[1], "OOS excess equity (S2 − SPY)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5 — monthly + losing months
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

        # Page 7 — health + attribution
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), facecolor=_BG)
        for ax in axes:
            ax.set_facecolor(_BG)
            ax.axis("off")
        if health is not None and not health.empty:
            axes[0].table(
                cellText=np.round(health.select_dtypes(include=[np.number]), 2).values
                if not health.empty
                else [],
                rowLabels=[f"{r.pair_id}/{r.window}" for r in health.itertuples()],
                colLabels=[
                    c for c in health.columns if c not in {"pair_id", "window"}
                ],
                loc="center",
            )
            axes[0].set_title("Reversion health", color=_NAVY, fontweight="bold")
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

        # Page 8 — trades + capacity
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
                    "z_entry",
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

        if capacity is not None and not capacity.empty:
            fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=_BG)
            ax.set_facecolor(_BG)
            ax.axis("off")
            show = capacity.copy()
            for c in show.columns:
                if np.issubdtype(show[c].dtype, np.number):
                    show[c] = show[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            ax.table(
                cellText=show.values,
                colLabels=list(show.columns),
                loc="center",
            )
            ax.set_title(
                "Capacity (20d ADV vs $1m book / 6 slots)",
                color=_NAVY,
                fontweight="bold",
            )
            pdf.savefig(fig)
            plt.close(fig)

        # Page 9 — tail / CI / gap
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), facecolor=_BG)
        axes[0].set_facecolor(_BG)
        axes[0].axis("off")
        txt = (
            f"CVaR 5% = {cvar_5:.4f}\n"
            f"Sharpe = {sharpe_ci.get('sharpe', float('nan')):.3f}\n"
            f"95% CI = [{sharpe_ci.get('ci_low', float('nan')):.3f}, "
            f"{sharpe_ci.get('ci_high', float('nan')):.3f}]\n"
            f"PSR = {headline.get('psr', float('nan')):.3f}\n"
            f"DSR stack = {headline.get('dsr_stack', float('nan')):.3f}"
        )
        axes[0].text(0.05, 0.8, txt, fontsize=12, family="monospace", va="top")
        axes[0].set_title("Tail & Sharpe inference (OOS)", color=_NAVY, fontweight="bold")
        axes[1].set_facecolor(_BG)
        if gap_table is not None and not gap_table.empty:
            order = ["session", "weekday_overnight", "weekend"]
            t = gap_table.set_index("bucket").reindex(order)
            axes[1].bar(
                order,
                [10000.0 * float(v) if np.isfinite(v) else 0.0 for v in t["mean_return"]],
                color=[_NAVY, _STEEL, _RED],
            )
            axes[1].axhline(0.0, color="gray", lw=0.8)
            _style_pdf_ax(axes[1], "Mean return by gap bucket (bps)")
        else:
            axes[1].axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return path
