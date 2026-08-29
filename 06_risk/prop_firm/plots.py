"""Plotly figures for prop-firm notebooks."""

from __future__ import annotations

import pandas as pd


def failure_mix_figure(mix: pd.Series, *, title: str = "First binding constraint"):
    import plotly.graph_objects as go

    s = mix.fillna("none")
    fig = go.Figure(go.Bar(x=[str(i) for i in s.index], y=s.to_numpy(), marker_color="#a01313"))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=360,
        xaxis_title="constraint",
        yaxis_title="failed paths",
    )
    return fig


def leverage_heatmap_figure(grid: pd.DataFrame, *, value: str = "ev_per_day"):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=grid["leverage"],
            y=grid[value],
            marker_color="#1f4e79",
            name=value,
        )
    )
    fig.update_layout(
        title=f"Leverage vs {value} (optimize EV, not pass rate)",
        template="plotly_white",
        height=380,
        xaxis_title="leverage k",
        yaxis_title=value,
    )
    return fig


def retries_hist_figure(attempts: pd.Series, *, title: str = "Attempts until first two-step pass"):
    import plotly.graph_objects as go

    fig = go.Figure(go.Histogram(x=attempts, nbinsx=30, marker_color="#5b7c99"))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=360,
        xaxis_title="attempts (geometric given p_both)",
        yaxis_title="count",
    )
    return fig
