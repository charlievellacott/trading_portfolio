"""Fold-val reporting for S2 research notebooks. Does not assign STAR."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from backtest.s2_coint.runner import fit_hmm_on_train_dates, run_s2_backtest
from backtest.s2_coint.walkforward import S2WalkForwardFold
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.metrics import corr_to_s1, metrics_from_returns


def fold_table(folds: list[S2WalkForwardFold]) -> pd.DataFrame:
    rows = []
    for f in folds:
        rows.append(
            {
                "fold_id": f.fold_id,
                "train_start": f.train_dates.min(),
                "train_end": f.train_dates.max(),
                "n_train": len(f.train_dates),
                "embargo_start": f.embargo_dates.min() if len(f.embargo_dates) else pd.NaT,
                "embargo_end": f.embargo_dates.max() if len(f.embargo_dates) else pd.NaT,
                "val_start": f.val_dates.min(),
                "val_end": f.val_dates.max(),
                "n_val": len(f.val_dates),
            }
        )
    return pd.DataFrame(rows)


def _fold_val_panel_stats(
    panel: pd.DataFrame,
    val_dates: pd.DatetimeIndex,
    *,
    bar: str,
    score_column: str,
) -> dict:
    """Val-window z coverage and half-life in bars and sessions."""
    from backtest.s2_coint.research import half_life_to_sessions

    dates = pd.to_datetime(panel["date"])
    val = panel.loc[dates.isin(val_dates)]
    if val.empty:
        return {
            "pct_z_finite": float("nan"),
            "median_hl_bars": float("nan"),
            "median_hl_sessions": float("nan"),
        }
    z = (
        val[score_column].astype(float)
        if score_column in val.columns
        else pd.Series(dtype=float)
    )
    hl = (
        val["half_life"].astype(float)
        if "half_life" in val.columns
        else pd.Series(dtype=float)
    )
    hl_fin = hl[np.isfinite(hl)]
    med_bars = float(hl_fin.median()) if len(hl_fin) else float("nan")
    med_sess = (
        float(half_life_to_sessions(med_bars, bar=bar)) if np.isfinite(med_bars) else float("nan")
    )
    return {
        "pct_z_finite": float(np.isfinite(z.to_numpy(dtype=float)).mean()) if len(z) else float("nan"),
        "median_hl_bars": med_bars,
        "median_hl_sessions": med_sess,
    }


def _fold_train_state(
    panel: pd.DataFrame,
    fold: S2WalkForwardFold,
    cfg: S2SimConfig,
    *,
    score_column: str,
) -> tuple[float, dict | None]:
    """Tier B fold-train fit: sizing scale and optional HMM (never pick knobs on train Sharpe)."""
    dates = pd.to_datetime(panel["date"])
    train_mask = dates.isin(fold.train_dates)
    mean_abs = 1.0
    if cfg.size_mode != "equal" and score_column in panel.columns:
        sl = panel.loc[train_mask, score_column]
        if len(sl):
            m = float(sl.astype(float).abs().mean())
            if np.isfinite(m) and m > 0:
                mean_abs = m
    hmm = None
    if cfg.entry_mode == "v3_hmm_innov":
        hmm = fit_hmm_on_train_dates(panel, fold.train_dates)
    return mean_abs, hmm


def fold_val_metrics(
    panel: pd.DataFrame,
    folds: list[S2WalkForwardFold],
    configs: dict[str, S2SimConfig],
    *,
    s1_weekly: pd.Series | None = None,
    score_column: str = "z",
) -> pd.DataFrame:
    """Score each named config on each fold's **validation** dates only.

    Tier B arms: ``mean_abs_score`` and HMM are fit on fold-train only; val is scored
    out-of-sample within the fold. Returns one row per arm × fold_id (full ``fold_df``).
    """
    rows = []
    dates = pd.to_datetime(panel["date"])
    for name, cfg in configs.items():
        for f in folds:
            mean_abs, hmm = _fold_train_state(panel, f, cfg, score_column=score_column)
            mask = dates.isin(f.val_dates)
            res = run_s2_backtest(
                panel,
                cfg,
                date_mask=mask,
                s1_weekly=s1_weekly,
                mean_abs_score=mean_abs,
                hmm_params=hmm,
            )
            n_entries = int(sum(pr.n_entries for pr in res.book.pair_results.values()))
            stats = _fold_val_panel_stats(
                panel, f.val_dates, bar=str(cfg.bar), score_column=score_column
            )
            rows.append(
                {
                    "arm": name,
                    "fold_id": f.fold_id,
                    "ann_sharpe": res.metrics["ann_sharpe"],
                    "max_drawdown": res.metrics["max_drawdown"],
                    "corr_to_s1": res.metrics["corr_to_s1"],
                    "n_days": res.metrics["n_days"],
                    "n_entries": n_entries,
                    "pct_z_finite": stats["pct_z_finite"],
                    "median_hl_bars": stats["median_hl_bars"],
                    "median_hl_sessions": stats["median_hl_sessions"],
                }
            )
    return pd.DataFrame(rows)


def full_is_metrics(
    panel: pd.DataFrame,
    configs: dict[str, S2SimConfig],
    *,
    s1_weekly: pd.Series | None = None,
    score_column: str = "z",
) -> pd.DataFrame:
    """One backtest per arm on the full research-IS panel (reporting only; not for STAR auto-pick).

    Tier B: sizing scale and HMM are fit on the entire ``panel`` IS window.
    """
    rows = []
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"])).sort_values().unique()
    for name, cfg in configs.items():
        mean_abs = 1.0
        if cfg.size_mode != "equal" and score_column in panel.columns:
            m = float(panel[score_column].astype(float).abs().mean())
            if np.isfinite(m) and m > 0:
                mean_abs = m
        hmm = (
            fit_hmm_on_train_dates(panel, dates)
            if cfg.entry_mode == "v3_hmm_innov"
            else None
        )
        res = run_s2_backtest(
            panel,
            cfg,
            s1_weekly=s1_weekly,
            mean_abs_score=mean_abs,
            hmm_params=hmm,
        )
        n_entries = int(sum(pr.n_entries for pr in res.book.pair_results.values()))
        rows.append(
            {
                "arm": name,
                "full_is_sharpe": res.metrics["ann_sharpe"],
                "full_is_max_drawdown": res.metrics["max_drawdown"],
                "full_is_corr_to_s1": res.metrics["corr_to_s1"],
                "full_is_n_days": res.metrics["n_days"],
                "full_is_n_entries": n_entries,
            }
        )
    return pd.DataFrame(rows)


def arm_selection_table(
    fold_df: pd.DataFrame,
    full_is_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join median val Sharpe, full-IS Sharpe, and val stability stats — no winner column."""
    if fold_df.empty:
        cols = [
            "arm",
            "median_val_sharpe",
            "full_is_sharpe",
            "max_drawdown",
            "corr_to_s1",
            "n_folds",
        ]
        return pd.DataFrame(columns=cols)
    val = (
        fold_df.groupby("arm", as_index=False)
        .agg(
            median_val_sharpe=("ann_sharpe", "median"),
            max_drawdown=("max_drawdown", "median"),
            corr_to_s1=("corr_to_s1", "median"),
            n_folds=("fold_id", "nunique"),
        )
    )
    if full_is_df is not None and not full_is_df.empty:
        val = val.merge(
            full_is_df[["arm", "full_is_sharpe"]],
            on="arm",
            how="left",
        )
    else:
        val["full_is_sharpe"] = float("nan")
    return val[
        [
            "arm",
            "median_val_sharpe",
            "full_is_sharpe",
            "max_drawdown",
            "corr_to_s1",
            "n_folds",
        ]
    ]


def median_sharpe_hint(fold_df: pd.DataFrame) -> str | None:
    """Commentary only — never assign a STAR variable from this."""
    if fold_df.empty or "ann_sharpe" not in fold_df.columns:
        return None
    med = fold_df.groupby("arm")["ann_sharpe"].median().dropna()
    if med.empty:
        return None
    best = med.idxmax()
    return str(best)


def plot_fold_boxplots(fold_df: pd.DataFrame, *, title: str = "") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, lab in zip(
        axes,
        ["ann_sharpe", "max_drawdown", "corr_to_s1"],
        ["Sharpe", "Max DD", "Corr to S1"],
    ):
        fold_df.boxplot(column=col, by="arm", ax=ax)
        ax.set_title(lab)
        ax.set_xlabel("")
    fig.suptitle(title or "Fold-val metrics (you type STAR after review)")
    plt.tight_layout()


def require_star(name: str, value) -> None:
    if value is None:
        raise ValueError(
            f"{name} is None. Type it in the freeze cell after reviewing fold-val metrics."
        )


def load_star_stack(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"star stack not found: {path}. "
            "Use DEFAULT_STAR_STACK from backtest.s2_coint.research "
            "(04_backtest/s2_coint/artifacts/s2_star_stack.json)."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_star_stack(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def write_tearsheet_pdf(path: str, returns: pd.Series, *, title: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    orig_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        with PdfPages(path) as pdf:
            fig, ax = plt.subplots(figsize=(8, 4))
            eq = (1.0 + returns.fillna(0.0)).cumprod()
            ax.plot(eq.index, eq.values)
            ax.set_title(title)
            ax.set_ylabel("equity")
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 3))
            m = metrics_from_returns(returns)
            ax.axis("off")
            ax.text(
                0.1,
                0.5,
                f"Sharpe={m['ann_sharpe']}\nmaxDD={m['max_drawdown']}\nn={m['n_days']}",
                fontsize=12,
            )
            pdf.savefig(fig)
            plt.close(fig)
    finally:
        plt.show = orig_show
