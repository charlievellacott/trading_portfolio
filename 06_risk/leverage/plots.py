"""Plotly bars for the leverage surface (CAGR / Calmar / drawdown vs target vol)."""

from __future__ import annotations

import pandas as pd


def _bar(grid: pd.DataFrame, *, y: str, title: str, yaxis_title: str):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=grid["target_ann_vol"],
            y=grid[y],
            marker_color="#1f4e79",
            name=y,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=380,
        xaxis_title="target_ann_vol",
        yaxis_title=yaxis_title,
    )
    return fig


def surface_cagr_figure(grid: pd.DataFrame, *, segment: str = "oos"):
    return _bar(
        grid,
        y=f"{segment}_cagr",
        title=f"{segment.upper()} CAGR vs target vol (not Sharpe)",
        yaxis_title="CAGR",
    )


def surface_calmar_figure(grid: pd.DataFrame, *, segment: str = "oos"):
    return _bar(
        grid,
        y=f"{segment}_calmar",
        title=f"{segment.upper()} Calmar vs target vol",
        yaxis_title="Calmar",
    )


def surface_dd_figure(grid: pd.DataFrame, *, segment: str = "oos"):
    return _bar(
        grid,
        y=f"{segment}_max_drawdown",
        title=f"{segment.upper()} max drawdown vs target vol",
        yaxis_title="max drawdown",
    )
