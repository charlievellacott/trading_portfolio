"""S3 research diagnostics: Sharpe / DD / Calmar, PSR/DSR, corr to S1 and S2."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from data.repo_paths import repo_root

PERIODS_PER_YEAR = 252.0


def _default_s1_paths() -> list[str]:
    root = repo_root()
    return [
        os.path.join(root, "01_data", "data_files", "s1_equities", "s1_period_returns.parquet"),
        os.path.join(root, "04_backtest", "s1_equities", "artifacts", "s1_period_returns.parquet"),
        os.path.join(root, "09_performance", "cache", "s1_period_returns.parquet"),
    ]


def _default_s2_paths() -> list[str]:
    root = repo_root()
    return [
        os.path.join(root, "04_backtest", "s2_coint", "artifacts", "s2_period_returns.parquet"),
        os.path.join(root, "01_data", "data_files", "s2_coint", "s2_period_returns.parquet"),
        os.path.join(root, "09_performance", "cache", "s2_period_returns.parquet"),
    ]


def metrics_from_returns(
    ret: pd.Series,
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> dict:
    """Ann. Sharpe, max drawdown, Calmar (CAGR/|MDD|), and observation count."""
    if ret is None or ret.empty:
        return {
            "ann_sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "n_days": 0,
        }
    r = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if r.empty:
        return {
            "ann_sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "n_days": 0,
        }
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    sharpe = float(np.sqrt(periods_per_year) * mu / sd) if sd > 0 else float("nan")
    equity = (1.0 + r).cumprod()
    mdd = float((equity / equity.cummax() - 1.0).min())
    n = int(r.shape[0])
    years = n / float(periods_per_year) if periods_per_year > 0 else float("nan")
    if years and years > 0 and np.isfinite(years):
        cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")
    if np.isfinite(mdd) and abs(mdd) > 1e-12 and np.isfinite(cagr):
        calmar = float(cagr / abs(mdd))
    else:
        calmar = float("nan")
    return {
        "ann_sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
        "n_days": n,
    }


def metrics_from_returns_inference(
    ret: pd.Series,
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
    n_trials_local: int | None = None,
    n_trials_stack: int | None = None,
    psr_benchmark: float = 0.0,
) -> dict:
    """Net Sharpe / DD / Calmar / n_days plus PSR and deflated Sharpe."""
    from performance.sharpe_inference import (
        deflated_sharpe_ratio,
        probabilistic_sharpe_ratio,
        return_moments,
    )

    base = metrics_from_returns(ret, periods_per_year=periods_per_year)
    out = {
        **base,
        "psr": float("nan"),
        "dsr_local": float("nan"),
        "dsr_stack": float("nan"),
        "skew": float("nan"),
        "excess_kurtosis": float("nan"),
        "n_trials_local": n_trials_local,
        "n_trials_stack": n_trials_stack,
    }
    if ret is None or ret.empty:
        return out
    mom = return_moments(ret, periods_per_year=periods_per_year)
    out["skew"] = mom["skew"]
    out["excess_kurtosis"] = (
        float(mom["kurtosis"] - 3.0) if np.isfinite(mom["kurtosis"]) else float("nan")
    )
    sr = mom["sr"]
    n = mom["n_obs"]
    sk = mom["skew"]
    ku = mom["kurtosis"]
    out["psr"] = probabilistic_sharpe_ratio(
        sr, n, sk, ku, sr_benchmark=psr_benchmark
    )
    if n_trials_local is not None and int(n_trials_local) >= 1:
        out["dsr_local"] = deflated_sharpe_ratio(sr, n, sk, ku, int(n_trials_local))
    if n_trials_stack is not None and int(n_trials_stack) >= 1:
        out["dsr_stack"] = deflated_sharpe_ratio(sr, n, sk, ku, int(n_trials_stack))
    return out


def cost_bps_per_year(
    ret: pd.Series,
    trades: pd.DataFrame,
    *,
    open_entry_cost_bps: float = 0.0,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> float:
    """Annualized cost drag in bps from completed (and open) fills."""
    n_days = int(ret.shape[0]) if ret is not None else 0
    if n_days <= 0:
        return float("nan")
    closed = 0.0
    if trades is not None and not trades.empty:
        entry = trades["entry_cost_bps"] if "entry_cost_bps" in trades.columns else 0.0
        exit_ = trades["exit_cost_bps"] if "exit_cost_bps" in trades.columns else 0.0
        closed = float(pd.to_numeric(entry, errors="coerce").fillna(0.0).sum())
        closed += float(pd.to_numeric(exit_, errors="coerce").fillna(0.0).sum())
        if "swap_cost_bps" in trades.columns:
            closed += float(
                pd.to_numeric(trades["swap_cost_bps"], errors="coerce").fillna(0.0).sum()
            )
    years = n_days / float(periods_per_year)
    if years <= 0:
        return float("nan")
    return (closed + float(open_entry_cost_bps)) / years


def compound_to_s1_weeks(daily: pd.Series, s1_index: pd.DatetimeIndex) -> pd.Series:
    """Compound daily net returns over each S1 hold ``[T, T_next)`` (Mon–Mon)."""
    if daily is None or daily.empty or len(s1_index) == 0:
        return pd.Series(dtype=float, name="s3_week")
    s = pd.to_numeric(daily, errors="coerce").dropna().astype(float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    weeks = pd.DatetimeIndex(pd.to_datetime(s1_index)).sort_values().unique()
    out: dict[pd.Timestamp, float] = {}
    for i, t in enumerate(weeks):
        t0 = pd.Timestamp(t)
        t1 = (
            pd.Timestamp(weeks[i + 1])
            if i + 1 < len(weeks)
            else t0 + pd.Timedelta(days=7)
        )
        window = s.loc[(s.index >= t0) & (s.index < t1)]
        if window.empty:
            continue
        out[t0] = float((1.0 + window).prod() - 1.0)
    return pd.Series(out, dtype=float, name="s3_week")


def corr_to_s1(daily: pd.Series, s1_weekly: pd.Series | None) -> float:
    """Pearson corr of Mon–Mon compounds vs S1 weekly returns. Sparse → NaN."""
    if daily is None or s1_weekly is None or daily.empty or s1_weekly.empty:
        return float("nan")
    s1 = pd.to_numeric(s1_weekly, errors="coerce").dropna().astype(float)
    s1.index = pd.to_datetime(s1.index)
    s1 = s1.sort_index()
    weekly = compound_to_s1_weeks(daily, s1.index)
    joined = pd.concat([weekly.rename("s3"), s1.rename("s1")], axis=1).dropna()
    if len(joined) < 3:
        return float("nan")
    return float(joined["s3"].corr(joined["s1"]))


def corr_to_s2(daily: pd.Series, s2_daily: pd.Series | None) -> float:
    """Pearson corr of daily returns vs S2 daily (aligned intersection)."""
    if daily is None or s2_daily is None or daily.empty or s2_daily.empty:
        return float("nan")
    a = pd.to_numeric(daily, errors="coerce").dropna().astype(float)
    a.index = pd.to_datetime(a.index)
    b = pd.to_numeric(s2_daily, errors="coerce").dropna().astype(float)
    b.index = pd.to_datetime(b.index)
    joined = pd.concat([a.rename("s3"), b.rename("s2")], axis=1).dropna()
    if len(joined) < 3:
        return float("nan")
    return float(joined["s3"].corr(joined["s2"]))


def _load_returns_parquet(path: str, name: str) -> pd.Series:
    if not path or not os.path.isfile(path):
        return pd.Series(dtype=float, name=name)
    df = pd.read_parquet(path)
    if isinstance(df, pd.Series):
        s = df
    elif "ret" in df.columns:
        s = df["ret"]
    else:
        s = df.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index)
    out = s.dropna().sort_index().astype(float)
    out.name = name
    return out


def load_s1_period_returns(path: str | None = None) -> pd.Series:
    """Weekly S1 net returns. Tries ``path`` then known artifact paths; missing → empty."""
    candidates: list[str] = []
    if path:
        candidates.append(path)
    candidates.extend(_default_s1_paths())
    for p in candidates:
        if os.path.isfile(p):
            return _load_returns_parquet(p, "s1")
    return pd.Series(dtype=float, name="s1")


def load_s2_period_returns(path: str | None = None) -> pd.Series:
    """Daily S2 net returns. Tries ``path`` then known artifact paths; missing → empty."""
    candidates: list[str] = []
    if path:
        candidates.append(path)
    candidates.extend(_default_s2_paths())
    for p in candidates:
        if os.path.isfile(p):
            return _load_returns_parquet(p, "s2")
    return pd.Series(dtype=float, name="s2")
