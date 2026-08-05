"""IS/OOS reporting helpers for S1 backtests."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from backtest.s1_equities.portfolio import (
    summarize_concentration,
    weight_concentration_stats,
)
from backtest.s1_equities.runner import BacktestResult, summarize_periods
from data.ingestion.equity_fetcher import fetch_ohlcv
from models.s1_equities.training_common import (
    LABEL_COL,
    date_ic_series,
    mean_date_ic,
)
from performance.metrics import (
    calmar_ratio,
    max_drawdown,
    rolling_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)

_IC_HORIZON_LABELS = ("fwd_ret_1", "fwd_ret_5", "fwd_ret_21")
_VOL_TRAIL = 26


def split_period_returns(
    period_returns: pd.Series,
    is_dates: pd.DatetimeIndex | pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Split period returns into IS / OOS by entry date."""
    is_set = set(pd.DatetimeIndex(pd.to_datetime(is_dates)))
    idx = pd.DatetimeIndex(period_returns.index)
    mask = pd.Series([d in is_set for d in idx], index=idx)
    return period_returns.loc[mask], period_returns.loc[~mask]


def equity_curve_from_returns(
    period_returns: pd.Series,
    *,
    start_value: float = 1.0,
) -> pd.Series:
    """
    Independent equity curve from period returns, starting at ``start_value``.

    Used for export so IS and OOS curves can each begin at the same level
    (OOS ignores any IS P&L).
    """
    r = period_returns.dropna().sort_index()
    if r.empty:
        return pd.Series(dtype=float, name="equity")
    start_idx = r.index[0] - pd.Timedelta(days=1)
    curve = pd.concat(
        [
            pd.Series([float(start_value)], index=[start_idx]),
            float(start_value) * (1.0 + r).cumprod(),
        ]
    )
    curve.name = "equity"
    return curve


def export_segment_equity_curves(
    period_returns: pd.Series,
    is_dates: pd.DatetimeIndex | pd.Series,
    *,
    artifacts_dir: str,
    stem: str,
    start_value: float = 1.0,
) -> tuple[str, str]:
    """
    Write separate IS and OOS equity CSVs, each starting at ``start_value``.

    Returns ``(is_path, oos_path)``.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    is_r, oos_r = split_period_returns(period_returns, is_dates)
    eq_is = equity_curve_from_returns(is_r, start_value=start_value)
    eq_oos = equity_curve_from_returns(oos_r, start_value=start_value)
    is_path = os.path.join(artifacts_dir, f"{stem}_equity_is.csv")
    oos_path = os.path.join(artifacts_dir, f"{stem}_equity_oos.csv")
    eq_is.to_frame().to_csv(is_path, index_label="date")
    eq_oos.to_frame().to_csv(oos_path, index_label="date")
    return is_path, oos_path


def metrics_table(
    period_returns: pd.Series,
    *,
    label: str,
    periods_per_year: float = 52.0,
) -> pd.Series:
    """Named metrics Series for one segment."""
    base = summarize_periods(period_returns, periods_per_year=periods_per_year)
    # Also try vectorbt daily-equivalent via expanding to equity then pct_change
    out = {
        "segment": label,
        **base,
    }
    r = period_returns.dropna()
    if len(r) >= 3:
        # map weekly returns onto a Series for performance.metrics (freq=w)
        try:
            out["sharpe_vbt"] = sharpe_ratio(r, freq="w", year_freq="52w")
            out["sortino_vbt"] = sortino_ratio(r, freq="w", year_freq="52w")
            out["calmar_vbt"] = calmar_ratio(r, freq="w", year_freq="52w")
            out["max_dd_vbt"] = max_drawdown(r, freq="w")
        except Exception:
            out["sharpe_vbt"] = np.nan
            out["sortino_vbt"] = np.nan
            out["calmar_vbt"] = np.nan
            out["max_dd_vbt"] = np.nan
    return pd.Series(out)


def compare_segments(
    result: BacktestResult,
    is_dates: pd.DatetimeIndex | pd.Series,
) -> pd.DataFrame:
    """IS vs OOS vs full metrics table."""
    is_r, oos_r = split_period_returns(result.period_returns, is_dates)
    rows = [
        metrics_table(is_r, label="IS"),
        metrics_table(oos_r, label="OOS"),
        metrics_table(result.period_returns, label="FULL"),
    ]
    return pd.DataFrame(rows).set_index("segment")


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_n_grid(grid_df: pd.DataFrame, *, title: str = "IS N grid (net of costs)"):
    """Sharpe / total return / max DD vs N."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(grid_df["n"], grid_df["sharpe"], marker="o")
    axes[0].set_title("IS Sharpe")
    axes[0].set_xlabel("N")
    axes[1].plot(grid_df["n"], grid_df["total_return"], marker="o")
    axes[1].set_title("IS total return")
    axes[1].set_xlabel("N")
    axes[2].plot(grid_df["n"], grid_df["max_drawdown"], marker="o")
    axes[2].set_title("IS max drawdown")
    axes[2].set_xlabel("N")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_equity_overlay(
    curves: dict[str, pd.Series],
    *,
    title: str,
    is_end: pd.Timestamp | None = None,
):
    """Overlay equity curves; optional vertical line at IS end."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, label=name)
    if is_end is not None:
        ax.axvline(pd.Timestamp(is_end), color="gray", ls="--", lw=1, label="IS end")
    ax.set_ylabel("Equity (start=1)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drawdown(equity: pd.Series, *, title: str = "Drawdown"):
    dd = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0.0, color="#a01313", alpha=0.35)
    ax.plot(dd.index, dd.values, color="#a01313", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(period_returns: pd.Series, *, window: int = 26, title: str = ""):
    r = period_returns.dropna()
    fig, ax = plt.subplots(figsize=(10, 3))
    if len(r) >= max(8, window // 2):
        try:
            roll = rolling_sharpe_ratio(r, window, freq="w", year_freq="52w")
            ax.plot(roll.index, roll.values)
        except Exception:
            # fallback manual
            mu = r.rolling(window, min_periods=max(8, window // 2)).mean() * 52
            sd = r.rolling(window, min_periods=max(8, window // 2)).std() * np.sqrt(52)
            roll = mu / sd.replace(0, np.nan)
            ax.plot(roll.index, roll.values)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_title(title or f"Rolling {window}-period Sharpe")
    fig.tight_layout()
    return fig


def plot_leverage_and_vol(
    result: BacktestResult,
    *,
    target_ann_vol: float | None = None,
    title: str = "Leverage and trailing vol estimate",
):
    """Twin-axis plot of leverage and ``vol_estimate`` (optional target line)."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    lev = result.leverage.dropna()
    vol = result.vol_estimate.dropna()
    if not lev.empty:
        ax.plot(lev.index, lev.values, color="#1f4e79", label="leverage")
        ax.set_ylabel("Leverage")
    ax2 = ax.twinx()
    if not vol.empty:
        ax2.plot(vol.index, vol.values, color="#a01313", alpha=0.75, label="vol_est")
        ax2.set_ylabel("Ann. vol estimate")
    if target_ann_vol is not None and np.isfinite(target_ann_vol):
        ax2.axhline(
            float(target_ann_vol),
            color="#a01313",
            ls="--",
            lw=0.9,
            label=f"target={target_ann_vol:g}",
        )
    ax.set_title(title)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    return fig


def calendar_year_table(
    period_returns: pd.Series,
    *,
    periods_per_year: float = 52.0,
) -> pd.DataFrame:
    """Per-calendar-year return / vol / Sharpe / max DD from weekly period returns."""
    r = period_returns.dropna().sort_index()
    if r.empty:
        return pd.DataFrame(
            columns=["n_periods", "total_return", "vol", "sharpe", "max_drawdown"]
        )
    r = r.copy()
    r.index = pd.DatetimeIndex(pd.to_datetime(r.index))
    rows: list[dict] = []
    for year, g in r.groupby(r.index.year):
        sm = summarize_periods(g, periods_per_year=periods_per_year)
        rows.append(
            {
                "year": int(year),
                "n_periods": sm["n_periods"],
                "total_return": sm["total_return"],
                "vol": sm["vol"],
                "sharpe": sm["sharpe"],
                "max_drawdown": sm["max_drawdown"],
            }
        )
    return pd.DataFrame(rows).set_index("year")


def drawdown_episodes(
    equity: pd.Series,
    *,
    top_n: int = 5,
    periods_per_week: float = 1.0,
) -> pd.DataFrame:
    """
    Largest drawdown episodes by depth.

    Columns: depth (negative), peak_date, trough_date, recovery_date (NaT if open),
    length_weeks (peak→trough; recovery span separately as recovery_weeks).
    """
    eq = equity.dropna().sort_index()
    if eq.empty or len(eq) < 2:
        return pd.DataFrame(
            columns=[
                "depth",
                "peak_date",
                "trough_date",
                "recovery_date",
                "length_weeks",
                "recovery_weeks",
            ]
        )
    eq = eq.copy()
    eq.index = pd.DatetimeIndex(pd.to_datetime(eq.index))
    peak = eq.cummax()
    dd = eq / peak - 1.0
    # Segment: in drawdown when dd < 0
    in_dd = dd < -1e-12
    episodes: list[dict] = []
    i = 0
    idx = eq.index
    while i < len(eq):
        if not bool(in_dd.iloc[i]):
            i += 1
            continue
        # peak is the last max before this stretch; use running peak at i
        start = i
        while i < len(eq) and bool(in_dd.iloc[i]):
            i += 1
        end = i - 1  # last index still in drawdown
        seg = dd.iloc[start : end + 1]
        trough_loc = int(seg.values.argmin())
        trough_i = start + trough_loc
        depth = float(seg.iloc[trough_loc])
        trough_date = idx[trough_i]
        peak_val = float(peak.iloc[trough_i])
        pre = eq.iloc[: trough_i + 1]
        peak_hits = pre[np.isclose(pre.astype(float), peak_val)]
        peak_date = peak_hits.index[-1] if len(peak_hits) else idx[start]
        recovery_date = pd.NaT
        recovery_weeks = np.nan
        if i < len(eq):
            recovery_date = idx[i]
            recovery_weeks = float(
                (pd.Timestamp(recovery_date) - pd.Timestamp(peak_date)).days / 7.0
            )
        length_weeks = float(
            (pd.Timestamp(trough_date) - pd.Timestamp(peak_date)).days / 7.0
        )
        episodes.append(
            {
                "depth": depth,
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_date": recovery_date,
                "length_weeks": length_weeks,
                "recovery_weeks": recovery_weeks,
            }
        )
    if not episodes:
        return pd.DataFrame(
            columns=[
                "depth",
                "peak_date",
                "trough_date",
                "recovery_date",
                "length_weeks",
                "recovery_weeks",
            ]
        )
    out = pd.DataFrame(episodes).sort_values("depth").head(top_n)
    return out.reset_index(drop=True)


def _table_figure(df: pd.DataFrame, *, title: str, figsize=(10, 3.5)):
    """Render a small DataFrame as a matplotlib table figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center")
        fig.tight_layout()
        return fig
    cell = df.copy()
    for c in cell.columns:
        cell[c] = cell[c].map(
            lambda x: (
                f"{x:.4f}"
                if isinstance(x, (float, np.floating)) and pd.notna(x)
                else ("" if pd.isna(x) else str(x))
            )
        )
    ax.table(
        cellText=cell.values,
        rowLabels=[str(i) for i in cell.index],
        colLabels=list(cell.columns),
        loc="center",
    )
    fig.tight_layout()
    return fig


def monthly_returns(period_returns: pd.Series) -> pd.Series:
    """Compound period (weekly) returns into calendar-month returns."""
    r = period_returns.dropna().sort_index()
    if r.empty:
        return pd.Series(dtype=float, name="monthly_return")
    idx = pd.DatetimeIndex(pd.to_datetime(r.index))
    r = r.copy()
    r.index = idx
    m = (1.0 + r).groupby([idx.year, idx.month]).prod() - 1.0
    m.index = pd.to_datetime([f"{y}-{mo:02d}-01" for y, mo in m.index])
    m.name = "monthly_return"
    return m.sort_index()


def losing_months_summary(
    period_returns: pd.Series,
    *,
    worst_n: int = 10,
) -> dict[str, pd.Series | pd.DataFrame]:
    """
    Worst months and yearly fraction of up months from period returns.

    Returns dict with ``monthly``, ``worst``, ``pct_months_up_by_year``.
    """
    m = monthly_returns(period_returns)
    if m.empty:
        empty = pd.Series(dtype=float)
        return {
            "monthly": m,
            "worst": empty.to_frame("monthly_return"),
            "pct_months_up_by_year": empty,
        }
    worst = m.sort_values().head(worst_n).to_frame("monthly_return")
    years = m.index.year
    pct_up = (m > 0).groupby(years).mean()
    pct_up.index.name = "year"
    pct_up.name = "pct_months_up"
    return {"monthly": m, "worst": worst, "pct_months_up_by_year": pct_up}


def plot_losing_months(monthly: pd.Series, *, title: str = "OOS monthly returns"):
    """Bar chart of monthly returns (red = loss)."""
    m = monthly.dropna()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    if m.empty:
        ax.set_title(title)
        fig.tight_layout()
        return fig
    colors = ["#a01313" if v < 0 else "#2a6f2a" for v in m.values]
    ax.bar(m.index, m.values, width=20, color=colors, align="center")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_ylabel("Month return")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def spy_weekly_open_returns(
    entry_dates: pd.DatetimeIndex | pd.Series,
    *,
    hold_days: int = 5,
) -> pd.Series:
    """
    SPY open-to-open returns aligned to strategy entry dates.

    For consecutive entries ``d_i``, ``d_{i+1}``: ``open[d_{i+1}] / open[d_i] - 1``.
    Last entry uses the open ``hold_days`` trading sessions later when available.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(entry_dates)).sort_values().unique()
    if len(dates) == 0:
        return pd.Series(dtype=float, name="spy_weekly")

    start = (dates.min() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=21)).strftime("%Y-%m-%d")
    spy = fetch_ohlcv("SPY", start, end)
    if spy is None or spy.empty or "open" not in spy.columns:
        return pd.Series(dtype=float, name="spy_weekly")

    opens = spy.set_index(pd.to_datetime(spy["date"]))["open"].astype(float).sort_index()
    opens = opens[~opens.index.duplicated(keep="last")]

    # Map each entry to the on-or-before SPY session open.
    entry_px = opens.reindex(dates, method="ffill")
    out = pd.Series(index=dates, dtype=float, name="spy_weekly")
    for i, d in enumerate(dates):
        e0 = entry_px.loc[d]
        if not np.isfinite(e0) or e0 <= 0:
            out.loc[d] = np.nan
            continue
        if i + 1 < len(dates):
            e1 = entry_px.loc[dates[i + 1]]
        else:
            # next hold_days trading sessions after d
            loc = opens.index.searchsorted(d, side="left")
            exit_loc = loc + int(hold_days)
            if exit_loc >= len(opens):
                out.loc[d] = np.nan
                continue
            e1 = float(opens.iloc[exit_loc])
        if not np.isfinite(e1) or e1 <= 0:
            out.loc[d] = np.nan
            continue
        out.loc[d] = float(e1 / e0 - 1.0)
    return out


def _regime_bucket_stats(returns: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Mean return / Sharpe / n / win rate by regime label."""
    rows: list[dict] = []
    aligned = pd.concat(
        [returns.rename("r"), labels.rename("regime")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        return pd.DataFrame(
            columns=["n", "mean_return", "total_return", "sharpe", "win_rate"]
        )
    for name, g in aligned.groupby("regime", sort=False):
        sm = summarize_periods(g["r"])
        rows.append(
            {
                "regime": name,
                "n": sm["n_periods"],
                "mean_return": sm["mean_return"],
                "total_return": sm["total_return"],
                "sharpe": sm["sharpe"],
                "win_rate": sm["win_rate"],
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def regime_performance(
    oos_returns: pd.Series,
    spy_weekly: pd.Series,
    is_dates: pd.DatetimeIndex | pd.Series,
    *,
    vol_window: int = _VOL_TRAIL,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    OOS strategy returns by market direction and by SPY vol tercile.

    Vol tercile cutpoints are fit on IS trailing vol only, then applied to OOS.
    """
    spy = spy_weekly.dropna().sort_index()
    spy.index = pd.DatetimeIndex(pd.to_datetime(spy.index))
    oos = oos_returns.dropna().sort_index()
    oos.index = pd.DatetimeIndex(pd.to_datetime(oos.index))

    market_dir = pd.Series(
        np.where(spy > 0, "Up", "Down"), index=spy.index, name="market"
    )
    market_tbl = _regime_bucket_stats(oos, market_dir.reindex(oos.index))

    trail_vol = spy.rolling(vol_window, min_periods=max(8, vol_window // 2)).std()
    is_set = set(pd.DatetimeIndex(pd.to_datetime(is_dates)))
    is_vol = trail_vol.loc[[d for d in trail_vol.index if d in is_set]].dropna()
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

    vol_labels = trail_vol.map(_vol_label)
    vol_labels.name = "vol_regime"
    vol_tbl = _regime_bucket_stats(oos, vol_labels.reindex(oos.index))

    return {
        "market": market_tbl,
        "vol": vol_tbl,
        "spy_weekly": spy,
        "trail_vol": trail_vol,
        "vol_cutpoints": pd.Series({"q33": q33, "q66": q66}),
    }


def turnover_drag_summary(
    result: BacktestResult,
    is_dates: pd.DatetimeIndex | pd.Series,
) -> dict[str, float | pd.DataFrame | pd.Series]:
    """OOS gross−net cost drag vs turnover diagnostics."""
    _, oos_net = split_period_returns(result.period_returns, is_dates)
    _, oos_gross = split_period_returns(result.period_returns_gross, is_dates)
    to = result.turnover.reindex(oos_net.index).astype(float)
    drag = (oos_gross - oos_net).rename("cost_drag")
    frame = pd.DataFrame(
        {
            "ret_net": oos_net,
            "ret_gross": oos_gross,
            "turnover": to,
            "cost_drag": drag,
        }
    ).dropna(how="all")
    corr = frame[["turnover", "cost_drag", "ret_net", "ret_gross"]].corr()
    net_sm = summarize_periods(oos_net)
    gross_sm = summarize_periods(oos_gross)
    mean_abs_gross = float(oos_gross.abs().mean()) if len(oos_gross) else np.nan
    mean_drag = float(drag.mean()) if len(drag.dropna()) else np.nan
    summary = pd.Series(
        {
            "mean_turnover": float(to.mean()) if len(to.dropna()) else np.nan,
            "median_turnover": float(to.median()) if len(to.dropna()) else np.nan,
            "mean_cost_drag": mean_drag,
            "mean_cost_over_abs_gross": (
                abs(mean_drag) / mean_abs_gross
                if mean_abs_gross and mean_abs_gross > 1e-12
                else np.nan
            ),
            "sharpe_net": net_sm["sharpe"],
            "sharpe_gross": gross_sm["sharpe"],
            "corr_turnover_drag": (
                float(frame["turnover"].corr(frame["cost_drag"]))
                if len(frame.dropna(subset=["turnover", "cost_drag"])) > 2
                else np.nan
            ),
        }
    )
    return {"frame": frame, "corr": corr, "summary": summary}


def plot_cost_drag_vs_turnover(
    frame: pd.DataFrame,
    *,
    title: str = "OOS cost drag vs turnover",
):
    """Scatter of period turnover vs gross−net drag."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = frame.dropna(subset=["turnover", "cost_drag"])
    if not sub.empty:
        ax.scatter(sub["turnover"], sub["cost_drag"], s=12, alpha=0.55)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("One-way turnover")
    ax.set_ylabel("Gross − net (cost drag)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def ic_stability_summary(
    preds: pd.DataFrame,
    is_dates: pd.DatetimeIndex | pd.Series,
    *,
    score_col: str = "score",
    label_col: str = LABEL_COL,
    roll_window: int = 26,
) -> dict[str, pd.Series | pd.DataFrame | dict]:
    """Per-date IC series, IS/OOS mean ICIR, and rolling mean IC."""
    if preds is None or preds.empty or score_col not in preds.columns:
        empty = pd.Series(dtype=float)
        return {
            "ics": empty,
            "ics_oos": empty,
            "rolling_mean": empty,
            "segment": pd.DataFrame(),
            "is_stats": {},
            "oos_stats": {},
        }
    if label_col not in preds.columns:
        empty = pd.Series(dtype=float)
        return {
            "ics": empty,
            "ics_oos": empty,
            "rolling_mean": empty,
            "segment": pd.DataFrame(),
            "is_stats": {},
            "oos_stats": {},
        }

    ics = date_ic_series(preds, score_col, label_col=label_col)
    ics.index = pd.DatetimeIndex(pd.to_datetime(ics.index))
    is_set = set(pd.DatetimeIndex(pd.to_datetime(is_dates)))
    is_mask = pd.Series([d in is_set for d in ics.index], index=ics.index)
    ics_is = ics.loc[is_mask]
    ics_oos = ics.loc[~is_mask]
    is_stats = mean_date_ic(
        preds.loc[preds["date"].isin(is_set)],
        score_col,
        label_col=label_col,
    )
    oos_stats = mean_date_ic(
        preds.loc[~preds["date"].isin(is_set)],
        score_col,
        label_col=label_col,
    )
    segment = pd.DataFrame({"IS": is_stats, "OOS": oos_stats}).T
    rolling_mean = ics_oos.rolling(
        roll_window, min_periods=max(8, roll_window // 2)
    ).mean()
    return {
        "ics": ics,
        "ics_oos": ics_oos,
        "rolling_mean": rolling_mean,
        "segment": segment,
        "is_stats": is_stats,
        "oos_stats": oos_stats,
    }


def ic_horizon_decay(
    preds: pd.DataFrame,
    features_path: str | None = None,
    *,
    score_col: str = "score",
    horizons: tuple[str, ...] = _IC_HORIZON_LABELS,
) -> pd.DataFrame:
    """
    Mean IC of ``score`` vs each forward-return horizon.

    Merges missing horizon columns from ``features_path`` when provided.
    """
    if preds is None or preds.empty or score_col not in preds.columns:
        return pd.DataFrame(columns=["mean_ic", "std_ic", "icir", "n"])

    frame = preds.copy()
    need = [h for h in horizons if h not in frame.columns]
    if need and features_path and os.path.isfile(features_path):
        cols = ["date", "ticker", *need]
        feat = pd.read_parquet(features_path, columns=cols)
        feat["date"] = pd.to_datetime(feat["date"])
        feat["ticker"] = feat["ticker"].astype(str).str.strip().str.upper()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
        frame = frame.merge(feat, on=["date", "ticker"], how="left")

    rows: list[dict] = []
    for h in horizons:
        if h not in frame.columns:
            rows.append(
                {"horizon": h, "mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan, "n": 0}
            )
            continue
        stats = mean_date_ic(frame, score_col, label_col=h)
        rows.append({"horizon": h, **stats})
    return pd.DataFrame(rows).set_index("horizon")


def plot_rolling_ic(
    ics: pd.Series,
    *,
    window: int = 26,
    title: str = "",
):
    """Rolling mean of per-date IC."""
    s = ics.dropna().sort_index()
    fig, ax = plt.subplots(figsize=(10, 3))
    if len(s) >= max(8, window // 2):
        roll = s.rolling(window, min_periods=max(8, window // 2)).mean()
        ax.plot(roll.index, roll.values)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_title(title or f"Rolling {window}-period mean IC")
    ax.set_ylabel("IC")
    fig.tight_layout()
    return fig


def concentration_vs_returns(
    entry_weights: pd.DataFrame,
    period_returns: pd.Series,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Weight concentration stats joined to period returns."""
    stats = weight_concentration_stats(entry_weights)
    r = period_returns.copy()
    r.index = pd.DatetimeIndex(pd.to_datetime(r.index))
    if not stats.empty:
        stats.index = pd.DatetimeIndex(pd.to_datetime(stats.index))
    joined = stats.join(r.rename("ret"), how="inner")
    summary = summarize_concentration(stats)
    corr_cols = [c for c in ["hhi", "max_abs_w", "ret"] if c in joined.columns]
    corr = joined[corr_cols].corr() if len(corr_cols) >= 2 else pd.DataFrame()
    return {"stats": stats, "joined": joined, "summary": summary, "corr": corr}


def plot_concentration(
    stats: pd.DataFrame,
    *,
    title: str = "OOS weight concentration",
):
    """Time series of max_|w| and HHI."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    if stats is not None and not stats.empty:
        if "max_abs_w" in stats.columns:
            axes[0].plot(stats.index, stats["max_abs_w"].values)
        axes[0].set_ylabel("max |w|")
        if "hhi" in stats.columns:
            axes[1].plot(stats.index, stats["hhi"].values)
        axes[1].set_ylabel("HHI")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


def write_tearsheet_pdf(
    path: str,
    *,
    result: BacktestResult,
    is_dates: pd.DatetimeIndex | pd.Series,
    metrics: pd.DataFrame,
    title: str,
    preds: pd.DataFrame | None = None,
    features_path: str | None = None,
    label_col: str = LABEL_COL,
) -> None:
    """
    Multi-page PDF: metrics, equity/risk overview, then underperformance diagnostics.

    Diagnostic pages fail soft (skip with a note) when SPY/labels are unavailable.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_r, oos_r = split_period_returns(result.period_returns, is_dates)
    is_end = pd.DatetimeIndex(pd.to_datetime(is_dates)).max()

    with PdfPages(path) as pdf:
        # page 1 — metrics table as figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.set_title(title)
        tbl = metrics.copy()
        display_cols = [
            c
            for c in [
                "n_periods",
                "total_return",
                "avg_ann_return",
                "cagr",
                "vol",
                "sharpe",
                "max_drawdown",
                "win_rate",
            ]
            if c in tbl.columns
        ]
        # Return / rate columns shown as percentages; Sharpe left as a ratio.
        pct_cols = {
            "total_return",
            "avg_ann_return",
            "cagr",
            "vol",
            "max_drawdown",
            "win_rate",
        }
        cell = tbl[display_cols].copy()
        for c in cell.columns:
            if c == "n_periods":
                cell[c] = cell[c].map(lambda x: f"{int(x)}" if pd.notna(x) else "")
            elif c in pct_cols:
                cell[c] = cell[c].map(
                    lambda x: f"{100.0 * x:.2f}%" if pd.notna(x) else ""
                )
            else:
                cell[c] = cell[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        ax.table(
            cellText=cell.values,
            rowLabels=list(cell.index),
            colLabels=list(cell.columns),
            loc="center",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_equity_overlay(
            {"net": result.equity, "gross": result.equity_gross},
            title=f"{title} — equity",
            is_end=is_end,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_drawdown(result.equity, title=f"{title} — drawdown (net)")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_rolling_sharpe(
            oos_r,
            title="Out-of-sample (OOS) rolling Sharpe (26 weeks)",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # --- Underperformance diagnostics ---
        try:
            lm = losing_months_summary(oos_r)
            fig = plot_losing_months(
                lm["monthly"], title="OOS monthly returns (losing months)"
            )
            pdf.savefig(fig)
            plt.close(fig)
            fig = _table_figure(lm["worst"], title="Worst OOS months")
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as exc:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis("off")
            ax.set_title(f"Losing months skipped: {exc}")
            pdf.savefig(fig)
            plt.close(fig)

        try:
            spy = spy_weekly_open_returns(result.period_returns.index)
            regimes = regime_performance(oos_r, spy, is_dates)
            fig = _table_figure(
                regimes["market"], title="OOS by SPY market direction (Up/Down)"
            )
            pdf.savefig(fig)
            plt.close(fig)
            fig = _table_figure(
                regimes["vol"],
                title="OOS by SPY vol tercile (IS cutpoints)",
            )
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as exc:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis("off")
            ax.set_title(f"Regimes skipped: {exc}")
            pdf.savefig(fig)
            plt.close(fig)

        try:
            if preds is not None and not preds.empty:
                ic = ic_stability_summary(
                    preds, is_dates, label_col=label_col
                )
                fig = plot_rolling_ic(
                    ic["ics_oos"],
                    title="OOS rolling mean IC (26 weeks)",
                )
                pdf.savefig(fig)
                plt.close(fig)
                fig = _table_figure(ic["segment"], title="IC IS vs OOS")
                pdf.savefig(fig)
                plt.close(fig)
                decay = ic_horizon_decay(preds, features_path)
                fig = _table_figure(decay, title="IC horizon decay")
                pdf.savefig(fig)
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.axis("off")
                ax.set_title("IC diagnostics skipped: no preds")
                pdf.savefig(fig)
                plt.close(fig)
        except Exception as exc:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis("off")
            ax.set_title(f"IC diagnostics skipped: {exc}")
            pdf.savefig(fig)
            plt.close(fig)

        try:
            drag = turnover_drag_summary(result, is_dates)
            fig = _table_figure(
                drag["summary"].to_frame("value"),
                title="OOS turnover drag summary",
            )
            pdf.savefig(fig)
            plt.close(fig)
            fig = plot_cost_drag_vs_turnover(drag["frame"])
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as exc:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis("off")
            ax.set_title(f"Turnover drag skipped: {exc}")
            pdf.savefig(fig)
            plt.close(fig)

        try:
            conc = concentration_vs_returns(
                result.entry_weights.reindex(oos_r.index),
                oos_r,
            )
            fig = plot_concentration(conc["stats"])
            pdf.savefig(fig)
            plt.close(fig)
            fig = _table_figure(
                conc["summary"].to_frame("value"),
                title="OOS weight concentration summary",
            )
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as exc:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis("off")
            ax.set_title(f"Concentration skipped: {exc}")
            pdf.savefig(fig)
            plt.close(fig)

