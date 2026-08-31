"""Load unlevered base period-return parquets (fail loudly if missing)."""

from __future__ import annotations

import os

import pandas as pd

from risk.monte_carlo.loaders import find_repo_root
from strategies.s2_coint.metrics import load_s1_period_returns

S1_BASE_EXPORT_NOTE = (
    "Export from 04_backtest/s1_equities/notebooks/08_oos_tearsheet.ipynb "
    "(unlevered period_returns_base). Run notebooks 01-07 to freeze stars, "
    "then 08 so s1_period_returns_base.parquet exists."
)
S2_BASE_EXPORT_NOTE = (
    "Export from 04_backtest/s2_coint/notebooks/01_star_tearsheet.ipynb "
    "(pre-VT daily book). Frozen s2_star_stack.json is required; the tearsheet "
    "writes s2_period_returns_base.parquet."
)


def require_base_parquet(path: str, hint: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Unlevered base period returns missing: {path}. {hint}"
        )


def s1_period_returns_base_path(repo_root: str) -> str:
    return os.path.join(
        repo_root,
        "01_data",
        "data_files",
        "s1_equities",
        "s1_period_returns_base.parquet",
    )


def s2_period_returns_base_path(repo_root: str) -> str:
    return os.path.join(
        repo_root,
        "01_data",
        "data_files",
        "s2_coint",
        "s2_period_returns_base.parquet",
    )


def _as_ret_series(path: str, hint: str, name: str) -> pd.Series:
    require_base_parquet(path, hint)
    df = pd.read_parquet(path)
    if "ret" in df.columns:
        s = df["ret"]
    else:
        s = df.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index)
    out = s.dropna().sort_index().astype(float)
    if out.empty:
        raise FileNotFoundError(
            f"Unlevered base period returns are empty: {path}. {hint}"
        )
    out.name = name
    return out


def load_s1_period_returns_base(repo_root: str | None = None) -> pd.Series:
    root = find_repo_root(repo_root)
    path = s1_period_returns_base_path(root)
    require_base_parquet(path, S1_BASE_EXPORT_NOTE)
    s = load_s1_period_returns(path)
    if s.empty:
        raise FileNotFoundError(
            f"Unlevered S1 base period returns are empty: {path}. {S1_BASE_EXPORT_NOTE}"
        )
    s.name = "base"
    return s


def load_s2_period_returns_base(repo_root: str | None = None) -> pd.Series:
    root = find_repo_root(repo_root)
    path = s2_period_returns_base_path(root)
    return _as_ret_series(path, S2_BASE_EXPORT_NOTE, "base")
