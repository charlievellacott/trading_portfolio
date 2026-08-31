"""CAGR / Calmar / CVaR surface over frozen-VT ``target_ann_vol``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.s1_equities.runner import summarize_periods
from risk.leverage.apply import cfg_with_target, overlay_vol_target
from risk.monte_carlo.ev_stats import cvar
from risk.s1_equities.vol_targeting import VolTargetConfig

DEFAULT_TARGETS = (0.06, 0.08, 0.10, 0.12, 0.15, 0.18)


def split_is_oos(
    returns: pd.Series,
    is_end: str | pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    """Split period returns at ``is_end`` (IS inclusive)."""
    r = pd.to_numeric(returns, errors="coerce").astype(float).dropna()
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    cut = pd.Timestamp(is_end)
    return r.loc[r.index <= cut], r.loc[r.index > cut]


def period_metrics(
    returns: pd.Series,
    *,
    periods_per_year: float,
    cvar_alpha: float = 0.05,
) -> dict[str, float]:
    """Sharpe, CAGR, max DD, Calmar, CVaR on one segment."""
    stats = summarize_periods(returns, periods_per_year=float(periods_per_year))
    dd = float(stats["max_drawdown"]) if stats["max_drawdown"] == stats["max_drawdown"] else float("nan")
    cagr = float(stats["cagr"]) if stats["cagr"] == stats["cagr"] else float("nan")
    if np.isfinite(cagr) and np.isfinite(dd) and abs(dd) > 1e-12:
        calmar = float(cagr / abs(dd))
    else:
        calmar = float("nan")
    r = pd.to_numeric(returns, errors="coerce").astype(float).dropna()
    return {
        "n_periods": float(stats["n_periods"]),
        "sharpe": float(stats["sharpe"]) if stats["sharpe"] == stats["sharpe"] else float("nan"),
        "cagr": cagr,
        "vol": float(stats["vol"]) if stats["vol"] == stats["vol"] else float("nan"),
        "max_drawdown": dd,
        "calmar": calmar,
        "cvar": cvar(r, alpha=cvar_alpha) if len(r) else float("nan"),
    }


def leverage_surface(
    base: pd.Series,
    cfg: VolTargetConfig,
    *,
    targets: list[float] | tuple[float, ...] = DEFAULT_TARGETS,
    is_end: str | pd.Timestamp,
    periods_per_year: float,
    cvar_alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per ``target_ann_vol`` with IS and OOS metrics (OOS is the desk pick)."""
    rows: list[dict[str, float]] = []
    for t in targets:
        cfg_t = cfg_with_target(cfg, float(t))
        levered = overlay_vol_target(base, cfg_t)
        is_r, oos_r = split_is_oos(levered, is_end)
        is_m = period_metrics(is_r, periods_per_year=periods_per_year, cvar_alpha=cvar_alpha)
        oos_m = period_metrics(oos_r, periods_per_year=periods_per_year, cvar_alpha=cvar_alpha)
        realized = period_metrics(
            levered, periods_per_year=periods_per_year, cvar_alpha=cvar_alpha
        )
        rows.append(
            {
                "target_ann_vol": float(t),
                "realized_vol": realized["vol"],
                "is_sharpe": is_m["sharpe"],
                "is_cagr": is_m["cagr"],
                "is_max_drawdown": is_m["max_drawdown"],
                "is_calmar": is_m["calmar"],
                "is_cvar": is_m["cvar"],
                "oos_sharpe": oos_m["sharpe"],
                "oos_cagr": oos_m["cagr"],
                "oos_max_drawdown": oos_m["max_drawdown"],
                "oos_calmar": oos_m["calmar"],
                "oos_cvar": oos_m["cvar"],
                "oos_n": oos_m["n_periods"],
                "is_n": is_m["n_periods"],
            }
        )
    return pd.DataFrame(rows)
