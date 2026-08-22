"""Asia C / locked-pair failure diagnosis (postmortem helpers for research notebooks)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from data.processing.feature_implementation.cointegration import COINT_PVALUE
from strategies.s2_coint.baseline import (
    ENTRY_Z,
    EXIT_Z,
    PERIODS_PER_YEAR,
    PairSimResult,
    simulate_pair_baseline,
)
from strategies.s2_coint.metrics import (
    cost_bps_per_year,
    metrics_from_returns,
    summarize_rolling_adf,
)

_ENRICHED_TRADE_COLS: tuple[str, ...] = (
    "pair_id",
    "side",
    "side_label",
    "entry_date",
    "exit_date",
    "hold_bars",
    "entry_cost_bps",
    "exit_cost_bps",
    "signal_date",
    "z_entry",
    "spread_entry",
    "pnl_pct",
)


def slice_panel_window(
    panel: pd.DataFrame,
    is_end: pd.Timestamp | str,
    use_oos: bool,
) -> pd.DataFrame:
    """Return research-IS rows only, or the full panel when ``use_oos`` is True."""
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    if use_oos:
        return out.sort_values(["pair_id", "date"]).reset_index(drop=True)
    end = pd.Timestamp(is_end)
    return (
        out.loc[out["date"] <= end]
        .sort_values(["pair_id", "date"])
        .reset_index(drop=True)
    )


def _signal_row_before_fill(g: pd.DataFrame, fill_date: pd.Timestamp) -> pd.Series | None:
    """Signal bar is the close-t row immediately before fill open t+1."""
    dates = pd.to_datetime(g["date"])
    prior = g.loc[dates < pd.Timestamp(fill_date)]
    if prior.empty:
        return None
    return prior.iloc[-1]


def _compound_pnl_pct(returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Compound daily net returns over [start, end] inclusive → percent of pair capital."""
    if returns is None or returns.empty:
        return float("nan")
    r = pd.to_numeric(returns, errors="coerce").astype(float)
    r.index = pd.to_datetime(r.index)
    window = r.loc[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
    finite = window[np.isfinite(window.to_numpy(dtype=float))]
    if finite.empty:
        return float("nan")
    return float((1.0 + finite).prod() - 1.0) * 100.0


def enrich_trades(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    pair_returns: pd.Series,
) -> pd.DataFrame:
    """Join blotter to panel signal bar; add z/spread at entry and net ``pnl_pct``."""
    empty = pd.DataFrame(columns=list(_ENRICHED_TRADE_COLS))
    if trades is None or trades.empty:
        return empty
    if panel is None or panel.empty:
        out = trades.copy()
        for col in _ENRICHED_TRADE_COLS:
            if col not in out.columns:
                out[col] = np.nan
        return out[list(_ENRICHED_TRADE_COLS)]

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    rows: list[dict] = []
    for row in trades.itertuples(index=False):
        entry_date = pd.Timestamp(row.entry_date)
        exit_date = pd.Timestamp(row.exit_date)
        side = int(row.side)
        pid = str(row.pair_id)
        g = p.loc[p["pair_id"].astype(str) == pid].sort_values("date")
        if g.empty:
            g = p.sort_values("date")
        signal = _signal_row_before_fill(g, entry_date)
        signal_date = (
            pd.Timestamp(signal["date"]) if signal is not None else pd.NaT
        )
        z_entry = (
            float(signal["z"])
            if signal is not None and np.isfinite(float(signal["z"]))
            else float("nan")
        )
        spread_entry = (
            float(signal["spread"])
            if signal is not None and "spread" in signal.index
            and np.isfinite(float(signal["spread"]))
            else float("nan")
        )
        rows.append(
            {
                "pair_id": pid,
                "side": side,
                "side_label": "long" if side > 0 else "short",
                "entry_date": entry_date,
                "exit_date": exit_date,
                "hold_bars": int(row.hold_bars),
                "entry_cost_bps": float(row.entry_cost_bps),
                "exit_cost_bps": float(row.exit_cost_bps),
                "signal_date": signal_date,
                "z_entry": z_entry,
                "spread_entry": spread_entry,
                "pnl_pct": _compound_pnl_pct(pair_returns, entry_date, exit_date),
            }
        )
    return pd.DataFrame(rows, columns=list(_ENRICHED_TRADE_COLS))


def gross_returns_from_net(
    net_returns: pd.Series,
    trades: pd.DataFrame,
    *,
    open_entry_cost_bps: float = 0.0,
) -> pd.Series:
    """Add closed-trip fill costs back onto daily net returns (gross before frictions).

    ``open_entry_cost_bps`` is accepted for API symmetry with the sim result but is
    not restored here (open fills have no exit row to anchor without extra metadata).
    """
    _ = open_entry_cost_bps
    if net_returns is None or net_returns.empty:
        return pd.Series(dtype=float, name="gross")
    gross = pd.to_numeric(net_returns, errors="coerce").astype(float).copy()
    gross.index = pd.to_datetime(gross.index)
    if trades is not None and not trades.empty:
        for row in trades.itertuples(index=False):
            ed = pd.Timestamp(row.entry_date)
            xd = pd.Timestamp(row.exit_date)
            if ed in gross.index:
                gross.loc[ed] = float(gross.loc[ed]) + float(row.entry_cost_bps) / 10_000.0
            if xd in gross.index:
                gross.loc[xd] = float(gross.loc[xd]) + float(row.exit_cost_bps) / 10_000.0
    gross.name = "gross"
    return gross


def _pct_days_abs_z_gt(g: pd.DataFrame, entry_z: float) -> float:
    if g.empty or "z" not in g.columns:
        return float("nan")
    z = pd.to_numeric(g["z"], errors="coerce")
    finite = z[np.isfinite(z.to_numpy(dtype=float))]
    if finite.empty:
        return float("nan")
    return float((finite.abs() > float(entry_z)).mean() * 100.0)


def _beta_drift_std(g: pd.DataFrame) -> float:
    if g.empty or "beta" not in g.columns:
        return float("nan")
    b = pd.to_numeric(g["beta"], errors="coerce")
    finite = b[np.isfinite(b.to_numpy(dtype=float))]
    if len(finite) < 2:
        return float("nan")
    return float(finite.std(ddof=1))


def _median_half_life(g: pd.DataFrame) -> float:
    if g.empty or "half_life" not in g.columns:
        return float("nan")
    hl = pd.to_numeric(g["half_life"], errors="coerce")
    finite = hl[np.isfinite(hl.to_numpy(dtype=float))]
    if finite.empty:
        return float("nan")
    return float(finite.median())


def failure_scorecard(
    panel: pd.DataFrame,
    *,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
    periods_per_year: float = PERIODS_PER_YEAR,
    adf_pvalue_threshold: float = COINT_PVALUE,
    gross: bool = True,
) -> pd.DataFrame:
    """Per-pair opportunity / edge / health table (net + optional gross Sharpe)."""
    cols = [
        "pair_id",
        "n_entries",
        "n_round_trips",
        "pct_days_abs_z_gt_entry",
        "median_hold_bars",
        "median_half_life",
        "ann_sharpe_net",
        "ann_sharpe_gross",
        "max_drawdown",
        "cost_bps_year",
        "median_adf_p",
        "last_adf_p",
        "pct_adf_lt_threshold",
        "beta_std",
        "n_days",
    ]
    if panel is None or panel.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for pair_id, g in panel.groupby("pair_id", sort=False):
        result: PairSimResult = simulate_pair_baseline(
            g, entry_z=entry_z, exit_z=exit_z
        )
        net_m = metrics_from_returns(
            result.returns, periods_per_year=periods_per_year
        )
        cost_yr = cost_bps_per_year(
            result.returns,
            result.trades,
            open_entry_cost_bps=result.open_entry_cost_bps,
            periods_per_year=periods_per_year,
        )
        adf = summarize_rolling_adf(g, pvalue_threshold=adf_pvalue_threshold)
        median_hold = (
            float(result.trades["hold_bars"].median())
            if not result.trades.empty
            else float("nan")
        )
        gross_sharpe = float("nan")
        if gross:
            g_ret = gross_returns_from_net(
                result.returns,
                result.trades,
                open_entry_cost_bps=result.open_entry_cost_bps,
            )
            gross_sharpe = metrics_from_returns(
                g_ret, periods_per_year=periods_per_year
            )["ann_sharpe"]
        rows.append(
            {
                "pair_id": str(pair_id),
                "n_entries": int(result.n_entries),
                "n_round_trips": int(len(result.trades)),
                "pct_days_abs_z_gt_entry": _pct_days_abs_z_gt(g, entry_z),
                "median_hold_bars": median_hold,
                "median_half_life": _median_half_life(g),
                "ann_sharpe_net": net_m["ann_sharpe"],
                "ann_sharpe_gross": gross_sharpe,
                "max_drawdown": net_m["max_drawdown"],
                "cost_bps_year": cost_yr,
                "median_adf_p": adf["median_adf_p"],
                "last_adf_p": adf["last_adf_p"],
                "pct_adf_lt_threshold": adf["pct_adf_lt_threshold"],
                "beta_std": _beta_drift_std(g),
                "n_days": net_m["n_days"],
            }
        )
    return pd.DataFrame(rows, columns=cols)


def simulate_and_enrich_panel(
    panel: pd.DataFrame,
    *,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Run baseline per pair; return enriched trades + net return series map."""
    all_trades: list[pd.DataFrame] = []
    returns_by_pair: dict[str, pd.Series] = {}
    if panel is None or panel.empty:
        return pd.DataFrame(columns=list(_ENRICHED_TRADE_COLS)), returns_by_pair
    for pair_id, g in panel.groupby("pair_id", sort=False):
        result = simulate_pair_baseline(g, entry_z=entry_z, exit_z=exit_z)
        returns_by_pair[str(pair_id)] = result.returns
        enriched = enrich_trades(result.trades, g, result.returns)
        if not enriched.empty:
            all_trades.append(enriched)
    if not all_trades:
        return pd.DataFrame(columns=list(_ENRICHED_TRADE_COLS)), returns_by_pair
    return pd.concat(all_trades, ignore_index=True), returns_by_pair


def extreme_trades(
    trades_enriched: pd.DataFrame,
    *,
    n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best and worst round-trips by net ``pnl_pct`` (pair-capital percent)."""
    cols = list(_ENRICHED_TRADE_COLS)
    empty = pd.DataFrame(columns=cols)
    if trades_enriched is None or trades_enriched.empty:
        return empty, empty
    t = trades_enriched.copy()
    t["pnl_pct"] = pd.to_numeric(t["pnl_pct"], errors="coerce")
    ranked = t.dropna(subset=["pnl_pct"]).sort_values("pnl_pct", ascending=False)
    if ranked.empty:
        return empty, empty
    best = ranked.head(int(n)).reset_index(drop=True)
    worst = ranked.tail(int(n)).sort_values("pnl_pct").reset_index(drop=True)
    return best, worst


def print_extreme_trades(trades_enriched: pd.DataFrame, n: int = 3) -> None:
    """Print best/worst trade dates for chart lookup."""
    best, worst = extreme_trades(trades_enriched, n=n)
    print(f"=== Best {n} trades by net pnl_pct ===")
    if best.empty:
        print("(none)")
    else:
        print(
            best[
                [
                    "pair_id",
                    "side_label",
                    "entry_date",
                    "exit_date",
                    "pnl_pct",
                    "z_entry",
                ]
            ].to_string(index=False)
        )
    print(f"\n=== Worst {n} trades by net pnl_pct ===")
    if worst.empty:
        print("(none)")
    else:
        print(
            worst[
                [
                    "pair_id",
                    "side_label",
                    "entry_date",
                    "exit_date",
                    "pnl_pct",
                    "z_entry",
                ]
            ].to_string(index=False)
        )


def _bollinger_on_spread(
    spread: pd.Series,
    *,
    window: int,
    entry_z: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    s = pd.to_numeric(spread, errors="coerce").astype(float)
    mid = s.rolling(int(window), min_periods=int(window)).mean()
    std = s.rolling(int(window), min_periods=int(window)).std(ddof=1)
    upper = mid + float(entry_z) * std
    lower = mid - float(entry_z) * std
    return mid, upper, lower


def plotly_pair_diagnosis(
    panel_g: pd.DataFrame,
    trades_enriched: pd.DataFrame,
    *,
    entry_z: float = ENTRY_Z,
    z_window: int = 60,
    pair_returns: pd.Series | None = None,
    is_end: pd.Timestamp | str | None = None,
    title: str | None = None,
):
    """Interactive spread + BB + trade markers, ADF, beta, cum net PnL."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if panel_g is None or panel_g.empty:
        fig = go.Figure()
        fig.update_layout(title=title or "empty panel")
        return fig

    g = panel_g.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(g["date"])
    spread = pd.to_numeric(g["spread"], errors="coerce")
    mid, upper, lower = _bollinger_on_spread(
        spread, window=z_window, entry_z=entry_z
    )
    adf = (
        pd.to_numeric(g["adf_pvalue"], errors="coerce")
        if "adf_pvalue" in g.columns
        else pd.Series(np.nan, index=g.index)
    )
    beta = (
        pd.to_numeric(g["beta"], errors="coerce")
        if "beta" in g.columns
        else pd.Series(np.nan, index=g.index)
    )

    if pair_returns is not None and not pair_returns.empty:
        r = pd.to_numeric(pair_returns, errors="coerce").astype(float)
        r.index = pd.to_datetime(r.index)
        # Align cum PnL to panel calendar (0 when flat).
        aligned = r.reindex(pd.DatetimeIndex(dates)).fillna(0.0)
        cum = (1.0 + aligned).cumprod() - 1.0
    else:
        cum = pd.Series(np.nan, index=g.index)

    pid = str(g["pair_id"].iloc[0]) if "pair_id" in g.columns else ""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(
            f"{pid} spread ± ENTRY_Z·σ",
            "rolling ADF p-value",
            "beta",
            "cumulative net PnL (pair capital)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=dates, y=spread, name="spread", line=dict(color="#1f77b4")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=mid, name="BB mid", line=dict(color="#7f7f7f", dash="dot")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=upper, name="BB upper", line=dict(color="#2ca02c", dash="dash")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=lower, name="BB lower", line=dict(color="#d62728", dash="dash")
        ),
        row=1,
        col=1,
    )

    if trades_enriched is not None and not trades_enriched.empty:
        t = trades_enriched.copy()
        t["entry_date"] = pd.to_datetime(t["entry_date"])
        t["exit_date"] = pd.to_datetime(t["exit_date"])
        spread_by_date = pd.Series(spread.to_numpy(), index=pd.DatetimeIndex(dates))

        for side, color, label in (
            (1, "#2ca02c", "long"),
            (-1, "#d62728", "short"),
        ):
            sub = t.loc[t["side"] == side]
            if sub.empty:
                continue
            entry_y = [
                float(spread_by_date.get(pd.Timestamp(d), np.nan))
                for d in sub["entry_date"]
            ]
            exit_y = [
                float(spread_by_date.get(pd.Timestamp(d), np.nan))
                for d in sub["exit_date"]
            ]
            fig.add_trace(
                go.Scatter(
                    x=list(sub["entry_date"]),
                    y=entry_y,
                    mode="markers",
                    name=f"{label} entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=10,
                        color=color,
                        line=dict(width=1, color="#000000"),
                    ),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(sub["exit_date"]),
                    y=exit_y,
                    mode="markers",
                    name=f"{label} exit",
                    marker=dict(
                        symbol="x",
                        size=9,
                        color=color,
                        line=dict(width=2, color=color),
                    ),
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(x=dates, y=adf, name="adf_pvalue", line=dict(color="#9467bd")),
        row=2,
        col=1,
    )
    fig.add_hline(y=float(COINT_PVALUE), line_dash="dot", row=2, col=1)
    fig.add_trace(
        go.Scatter(x=dates, y=beta, name="beta", line=dict(color="#8c564b")),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cum,
            name="cum_net_pnl",
            line=dict(color="#17becf"),
        ),
        row=4,
        col=1,
    )

    if is_end is not None:
        fig.add_vline(
            x=pd.Timestamp(is_end),
            line_dash="dash",
            line_color="#333333",
            annotation_text="IS end",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title or f"{pid} diagnosis",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="spread", row=1, col=1)
    fig.update_yaxes(title_text="ADF p", row=2, col=1)
    fig.update_yaxes(title_text="beta", row=3, col=1)
    fig.update_yaxes(title_text="cum ret", row=4, col=1)
    return fig


def check_fill_timing(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Light contract checks: signal close t → fill open t+1 (no same-bar close fill).

    Returns one row per trade with boolean flags. Empty trades → empty frame.
    """
    cols = [
        "pair_id",
        "entry_date",
        "signal_date",
        "ok_signal_before_fill",
        "ok_fill_has_open",
        "ok_not_same_bar_close_fill",
        "ok_z_finite_on_signal",
        "all_ok",
    ]
    if trades is None or trades.empty or panel is None or panel.empty:
        return pd.DataFrame(columns=cols)

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    rows: list[dict] = []
    for row in trades.itertuples(index=False):
        entry_date = pd.Timestamp(row.entry_date)
        pid = str(getattr(row, "pair_id", ""))
        g = p.loc[p["pair_id"].astype(str) == pid].sort_values("date")
        if g.empty:
            g = p.sort_values("date")
        date_set = set(pd.DatetimeIndex(g["date"]))
        signal = _signal_row_before_fill(g, entry_date)
        signal_date = (
            pd.Timestamp(signal["date"]) if signal is not None else pd.NaT
        )
        ok_signal = signal is not None and signal_date < entry_date
        fill_rows = g.loc[g["date"] == entry_date]
        ok_open = (
            not fill_rows.empty
            and np.isfinite(float(fill_rows.iloc[0]["open_y"]))
            and np.isfinite(float(fill_rows.iloc[0]["open_x"]))
            and entry_date in date_set
        )
        ok_not_same = bool(ok_signal and signal_date != entry_date)
        ok_z = signal is not None and np.isfinite(float(signal["z"]))
        all_ok = bool(ok_signal and ok_open and ok_not_same and ok_z)
        rows.append(
            {
                "pair_id": pid,
                "entry_date": entry_date,
                "signal_date": signal_date,
                "ok_signal_before_fill": bool(ok_signal),
                "ok_fill_has_open": bool(ok_open),
                "ok_not_same_bar_close_fill": bool(ok_not_same),
                "ok_z_finite_on_signal": bool(ok_z),
                "all_ok": all_ok,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def filter_pairs(panel: pd.DataFrame, pair_ids: Sequence[str]) -> pd.DataFrame:
    keep = set(pair_ids)
    if panel.empty:
        return panel.copy()
    return panel.loc[panel["pair_id"].isin(keep)].copy()
