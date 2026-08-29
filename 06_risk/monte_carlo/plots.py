"""Plotly figures for EV vs SPY notebooks."""

from __future__ import annotations

import pandas as pd

from risk.monte_carlo.ev_stats import path_wealth, terminal_simple_return


def equity_fan_figure(
    strategy_paths: pd.DataFrame,
    spy_paths: pd.DataFrame,
    *,
    title: str = "Equity fans (strategy vs SPY)",
    n_sample: int = 12,
):
    """Median / 5–95% bands plus a few sample paths."""
    import plotly.graph_objects as go

    w_s = path_wealth(strategy_paths)
    w_b = path_wealth(spy_paths)
    x = list(range(w_s.shape[0]))
    fig = go.Figure()
    for label, w, color in (
        ("strategy", w_s, "#1f4e79"),
        ("SPY", w_b, "#5b7c99"),
    ):
        q05 = w.quantile(0.05, axis=1)
        q50 = w.quantile(0.50, axis=1)
        q95 = w.quantile(0.95, axis=1)
        fig.add_trace(
            go.Scatter(
                x=list(x) + list(x[::-1]),
                y=list(q95) + list(q05[::-1]),
                fill="toself",
                fillcolor="rgba(31,78,121,0.12)" if label == "strategy" else "rgba(91,124,153,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{label} 5–95%",
                hoverinfo="skip",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q50,
                mode="lines",
                name=f"{label} median",
                line=dict(color=color, width=2),
            )
        )
        cols = list(w.columns[: max(0, int(n_sample))])
        for i, col in enumerate(cols):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=w[col],
                    mode="lines",
                    line=dict(color=color, width=0.6),
                    opacity=0.25,
                    showlegend=False,
                    name=f"{label} path {i}",
                )
            )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=440,
        hovermode="x unified",
        xaxis_title="bar",
        yaxis_title="wealth (start = 1)",
    )
    return fig


def terminal_hist_figure(
    strategy_paths: pd.DataFrame,
    spy_paths: pd.DataFrame,
    *,
    title: str = "Terminal simple return",
):
    import plotly.graph_objects as go

    ts = terminal_simple_return(strategy_paths)
    tb = terminal_simple_return(spy_paths)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ts, name="strategy", opacity=0.65, nbinsx=40))
    fig.add_trace(go.Histogram(x=tb, name="SPY", opacity=0.55, nbinsx=40))
    fig.update_layout(
        title=title,
        barmode="overlay",
        template="plotly_white",
        height=380,
        xaxis_title="W_H - 1",
        yaxis_title="paths",
    )
    return fig


def significance_frame(sig: pd.Series) -> pd.DataFrame:
    """One-row desk table for the EV-significance headline."""
    keys = [
        "mean",
        "mean_ann",
        "t_stat",
        "p_value",
        "ci_low",
        "ci_high",
        "ci_excludes_zero",
        "bootstrap_p_mean_le_0",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "n_obs",
        "psr",
        "psr_vs_1",
    ]
    row = {k: sig[k] for k in keys if k in sig.index}
    return pd.DataFrame([row])
