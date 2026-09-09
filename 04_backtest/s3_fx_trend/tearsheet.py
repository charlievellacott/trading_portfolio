"""Basic S3 tearsheet PDF (equity curve + metrics table)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from backtest.s3_fx_trend.runner import S3BacktestResult
from strategies.s3_fx_trend.metrics import metrics_from_returns_inference


def equity_from_returns(returns: pd.Series, *, start_value: float = 1.0) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    if r.empty:
        return pd.Series(dtype=float, name="equity")
    eq = float(start_value) * (1.0 + r).cumprod()
    eq.name = "equity"
    return eq


def write_tearsheet_pdf(
    result: S3BacktestResult | pd.Series,
    path: str,
    *,
    title: str = "S3 FX trend",
    n_trials_local: int | None = None,
    n_trials_stack: int | None = None,
    sleeve: str = "core",
) -> str:
    """Write a minimal PdfPages tearsheet: equity curve + metrics table."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _ = sleeve

    if isinstance(result, S3BacktestResult):
        returns = result.returns
        metrics = dict(result.metrics)
        if n_trials_local is None:
            n_trials_local = metrics.get("n_trials_local")
        if n_trials_stack is None:
            n_trials_stack = metrics.get("n_trials_stack")
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
