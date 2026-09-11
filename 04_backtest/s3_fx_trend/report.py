"""Fold-val reporting for S3 FX research notebooks. Does not assign STAR."""

from __future__ import annotations

import pandas as pd

from backtest.s3_fx_trend.runner import S3BacktestResult, run_s3_backtest
from backtest.s3_fx_trend.walkforward import S3WalkForwardFold
from backtest.star_stack_io import (
    load_star_stack,
    require_star,
    save_star_stack,
    update_star_stack_key,
    vt_target_ann_vol_from_stack,
)
from strategies.s3_fx_trend.config import S3SimConfig

# Re-exports for hypothesis notebooks.
__all__ = [
    "S3BacktestResult",
    "arm_selection_table",
    "fold_table",
    "fold_val_metrics",
    "full_is_metrics",
    "load_star_stack",
    "median_sharpe_hint",
    "require_star",
    "run_s3_backtest",
    "save_star_stack",
    "update_star_stack_key",
    "vt_target_ann_vol_from_stack",
]

_METRIC_KEYS = (
    "ann_sharpe",
    "ann_sharpe_gross",
    "max_drawdown",
    "calmar",
    "corr_to_s1",
    "corr_to_s2",
    "psr",
    "dsr_local",
    "dsr_stack",
    "skew",
    "excess_kurtosis",
    "cost_bps_per_year",
    "n_days",
    "n_entries",
    "n_trials_local",
    "n_trials_stack",
)


def fold_table(folds: list[S3WalkForwardFold]) -> pd.DataFrame:
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


def _inference_trial_counts(
    hyp_id: str | None,
    configs: dict[str, S3SimConfig],
    *,
    sleeve: str = "core",
) -> tuple[int | None, int | None]:
    if hyp_id is None:
        return None, None
    from backtest.s3_fx_trend.research import n_trials_local, n_trials_stack

    n_loc = n_trials_local(configs)
    n_stk = n_trials_stack(hyp_id, configs, sleeve=sleeve)
    return n_loc, n_stk


def _row_from_result(name: str, res: S3BacktestResult, *, fold_id: int | None = None) -> dict:
    row = {"arm": name}
    if fold_id is not None:
        row["fold_id"] = fold_id
    for k in _METRIC_KEYS:
        row[k] = res.metrics.get(k, float("nan") if k != "n_entries" else 0)
    return row


def fold_val_metrics(
    panel: pd.DataFrame,
    folds: list[S3WalkForwardFold],
    configs: dict[str, S3SimConfig],
    *,
    hyp_id: str | None = None,
    sleeve: str = "core",
    s1_weekly: pd.Series | None = None,
    s2_daily: pd.Series | None = None,
    rates_df: pd.DataFrame | None = None,
    reer_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score each named config on each fold's validation dates only."""
    rows = []
    dates = pd.to_datetime(panel["date"])
    n_loc, n_stk = _inference_trial_counts(hyp_id, configs, sleeve=sleeve)
    for name, cfg in configs.items():
        for f in folds:
            mask = dates.isin(f.val_dates)
            res = run_s3_backtest(
                panel,
                cfg,
                date_mask=mask,
                s1_weekly=s1_weekly,
                s2_daily=s2_daily,
                rates_df=rates_df,
                reer_df=reer_df,
                n_trials_local=n_loc,
                n_trials_stack=n_stk,
            )
            rows.append(_row_from_result(name, res, fold_id=f.fold_id))
    return pd.DataFrame(rows)


def full_is_metrics(
    panel: pd.DataFrame,
    configs: dict[str, S3SimConfig],
    *,
    hyp_id: str | None = None,
    sleeve: str = "core",
    s1_weekly: pd.Series | None = None,
    s2_daily: pd.Series | None = None,
    rates_df: pd.DataFrame | None = None,
    reer_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One backtest per arm on the full research-IS panel (reporting only)."""
    rows = []
    n_loc, n_stk = _inference_trial_counts(hyp_id, configs, sleeve=sleeve)
    for name, cfg in configs.items():
        res = run_s3_backtest(
            panel,
            cfg,
            s1_weekly=s1_weekly,
            s2_daily=s2_daily,
            rates_df=rates_df,
            reer_df=reer_df,
            n_trials_local=n_loc,
            n_trials_stack=n_stk,
        )
        row = _row_from_result(name, res)
        # Full-IS display aliases kept beside raw keys for notebook tables.
        row["full_is_sharpe"] = row["ann_sharpe"]
        row["full_is_ann_sharpe_gross"] = row["ann_sharpe_gross"]
        row["full_is_max_drawdown"] = row["max_drawdown"]
        row["full_is_corr_to_s1"] = row["corr_to_s1"]
        row["full_is_corr_to_s2"] = row["corr_to_s2"]
        rows.append(row)
    return pd.DataFrame(rows)


def arm_selection_table(
    fold_df: pd.DataFrame,
    full_is_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join median val Sharpe, full-IS Sharpe, and co-primaries — no winner column."""
    if fold_df.empty:
        cols = [
            "arm",
            "median_val_sharpe",
            "full_is_sharpe",
            "ann_sharpe_gross",
            "max_drawdown",
            "corr_to_s1",
            "corr_to_s2",
            "calmar",
            "median_psr",
            "median_dsr_local",
            "median_dsr_stack",
            "n_folds",
        ]
        return pd.DataFrame(columns=cols)

    agg_spec: dict[str, tuple[str, str]] = {
        "median_val_sharpe": ("ann_sharpe", "median"),
        "ann_sharpe_gross": ("ann_sharpe_gross", "median"),
        "max_drawdown": ("max_drawdown", "median"),
        "corr_to_s1": ("corr_to_s1", "median"),
        "n_folds": ("fold_id", "nunique"),
    }
    if "corr_to_s2" in fold_df.columns:
        agg_spec["corr_to_s2"] = ("corr_to_s2", "median")
    if "calmar" in fold_df.columns:
        agg_spec["calmar"] = ("calmar", "median")
    if "psr" in fold_df.columns:
        agg_spec["median_psr"] = ("psr", "median")
        agg_spec["median_dsr_local"] = ("dsr_local", "median")
        agg_spec["median_dsr_stack"] = ("dsr_stack", "median")
    val = fold_df.groupby("arm", as_index=False).agg(**agg_spec)

    if full_is_df is not None and not full_is_df.empty:
        merge_cols = ["arm"]
        sharpe_col = (
            "full_is_sharpe" if "full_is_sharpe" in full_is_df.columns else "ann_sharpe"
        )
        merge_cols.append(sharpe_col)
        for c in (
            "ann_sharpe_gross",
            "full_is_ann_sharpe_gross",
            "full_is_psr",
            "full_is_dsr_local",
            "full_is_dsr_stack",
            "psr",
            "dsr_local",
            "dsr_stack",
        ):
            if c in full_is_df.columns and c not in merge_cols:
                merge_cols.append(c)
        merged = full_is_df[merge_cols].copy()
        if sharpe_col != "full_is_sharpe":
            merged = merged.rename(columns={sharpe_col: "full_is_sharpe"})
        val = val.merge(merged, on="arm", how="left", suffixes=("", "_full"))
    else:
        val["full_is_sharpe"] = float("nan")

    out_cols = [
        "arm",
        "median_val_sharpe",
        "full_is_sharpe",
        "ann_sharpe_gross",
        "max_drawdown",
        "corr_to_s1",
        "n_folds",
    ]
    for c in (
        "corr_to_s2",
        "calmar",
        "median_psr",
        "median_dsr_local",
        "median_dsr_stack",
    ):
        if c in val.columns:
            out_cols.append(c)
    return val[out_cols]


def median_sharpe_hint(fold_df: pd.DataFrame) -> str | None:
    """Commentary only — never assign a STAR variable from this."""
    if fold_df.empty or "ann_sharpe" not in fold_df.columns:
        return None
    med = fold_df.groupby("arm")["ann_sharpe"].median().dropna()
    if med.empty:
        return None
    return str(med.idxmax())
