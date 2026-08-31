"""Plotly figures for EV vs SPY notebooks."""

from __future__ import annotations

import pandas as pd

from risk.analytics.monte_carlo.ev_stats import path_wealth, terminal_simple_return


def _add_fan_band(fig, x, w: pd.DataFrame, *, label: str, color: str, fill) -> None:
    import plotly.graph_objects as go

    q05 = w.quantile(0.05, axis=1)
    q50 = w.quantile(0.50, axis=1)
    q95 = w.quantile(0.95, axis=1)
    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(q95) + list(q05[::-1]),
            fill="toself",
            fillcolor=fill,
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


def equity_fan_figure(
    strategy_paths: pd.DataFrame,
    spy_paths: pd.DataFrame,
    *,
    title: str = "Equity fans (strategy vs SPY)",
    n_sample: int = 12,
    realized_strategy: pd.Series | None = None,
    realized_spy: pd.Series | None = None,
    realized_strategy_first: pd.Series | None = None,
    realized_spy_first: pd.Series | None = None,
):
    """Median / 5–95% bands, sample paths, and sealed OOS overlay."""
    import plotly.graph_objects as go

    w_s = path_wealth(strategy_paths)
    w_b = path_wealth(spy_paths)
    x = list(range(w_s.shape[0]))
    fig = go.Figure()
    _add_fan_band(
        fig, x, w_s, label="strategy", color="#1f4e79", fill="rgba(31,78,121,0.12)"
    )
    _add_fan_band(
        fig, x, w_b, label="SPY", color="#5b7c99", fill="rgba(91,124,153,0.12)"
    )
    cols = list(w_s.columns[: max(0, int(n_sample))])
    for i, col in enumerate(cols):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=w_s[col],
                mode="lines",
                line=dict(color="#1f4e79", width=0.6),
                opacity=0.22,
                showlegend=False,
                name=f"strategy path {i}",
            )
        )
    if realized_strategy is not None and len(realized_strategy):
        fig.add_trace(
            go.Scatter(
                x=list(realized_strategy.index),
                y=list(realized_strategy.values),
                mode="lines",
                name="OOS strategy (last H)",
                line=dict(color="#c4a35a", width=2.6),
            )
        )
    if realized_spy is not None and len(realized_spy):
        fig.add_trace(
            go.Scatter(
                x=list(realized_spy.index),
                y=list(realized_spy.values),
                mode="lines",
                name="OOS SPY (last H)",
                line=dict(color="#a01313", width=2.0, dash="dot"),
            )
        )
    if realized_strategy_first is not None and len(realized_strategy_first):
        fig.add_trace(
            go.Scatter(
                x=list(realized_strategy_first.index),
                y=list(realized_strategy_first.values),
                mode="lines",
                name="OOS strategy (first H)",
                line=dict(color="#c4a35a", width=1.6, dash="dash"),
            )
        )
    if realized_spy_first is not None and len(realized_spy_first):
        fig.add_trace(
            go.Scatter(
                x=list(realized_spy_first.index),
                y=list(realized_spy_first.values),
                mode="lines",
                name="OOS SPY (first H)",
                line=dict(color="#a01313", width=1.2, dash="dash"),
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


def excess_wealth_fan_figure(
    excess_wealth: pd.DataFrame,
    *,
    realized_excess: pd.Series | None = None,
    title: str = "Excess wealth (strategy / SPY)",
):
    """Relative-wealth fan so underperformance is not hidden by two cones."""
    import plotly.graph_objects as go

    x = list(range(excess_wealth.shape[0]))
    fig = go.Figure()
    _add_fan_band(
        fig,
        x,
        excess_wealth,
        label="W_s / W_spy",
        color="#1f4e79",
        fill="rgba(31,78,121,0.14)",
    )
    fig.add_hline(y=1.0, line_color="#8a9199", line_width=1)
    if realized_excess is not None and len(realized_excess):
        fig.add_trace(
            go.Scatter(
                x=list(realized_excess.index),
                y=list(realized_excess.values),
                mode="lines",
                name="OOS excess (last H)",
                line=dict(color="#c4a35a", width=2.6),
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400,
        hovermode="x unified",
        xaxis_title="bar",
        yaxis_title="W_strategy / W_SPY",
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


def max_dd_hist_figure(max_dd: pd.Series, *, title: str = "Pathwise max drawdown"):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Histogram(x=max_dd.astype(float), nbinsx=36, marker_color="#a01313", name="max DD")
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=360,
        xaxis_title="max drawdown (negative)",
        yaxis_title="paths",
    )
    return fig


def terminal_vs_dd_scatter_figure(
    holes: pd.DataFrame,
    *,
    title: str = "Terminal wealth vs max drawdown",
):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=holes["max_dd"],
            y=holes["terminal_wealth"],
            mode="markers",
            marker=dict(size=7, color="#1f4e79", opacity=0.55),
            name="paths",
        )
    )
    fig.add_hline(y=1.0, line_color="#8a9199", line_width=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400,
        xaxis_title="max drawdown",
        yaxis_title="terminal wealth",
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
