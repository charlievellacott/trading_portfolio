"""Half-Kelly ceiling, drawdown veto, Calmar/CAGR pick among survivors."""

from __future__ import annotations

import numpy as np
import pandas as pd

PICK_CALMAR = "calmar"
PICK_CAGR = "cagr"
VALID_PICK = frozenset({PICK_CALMAR, PICK_CAGR})


def half_kelly_target_vol(
    base: pd.Series,
    *,
    periods_per_year: float,
) -> float:
    """Equivalent annualized vol of half-Kelly on unlevered simple returns.

    Full Kelly fraction is ``mu / sigma^2`` (per-period). Half-Kelly is half of
    that. The matching vol-target is ``k_half * sigma_ann``. Non-positive mean
    or zero vol → 0 (no leverage above the floor).
    """
    r = pd.to_numeric(base, errors="coerce").astype(float).dropna()
    if len(r) < 3:
        return 0.0
    mu = float(r.mean())
    sig = float(r.std(ddof=1))
    if not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0.0 or mu <= 0.0:
        return 0.0
    k_half = 0.5 * mu / (sig ** 2)
    return float(k_half * sig * np.sqrt(float(periods_per_year)))


def apply_policy(
    surface: pd.DataFrame,
    *,
    half_kelly_vol: float,
    max_oos_dd: float = 0.25,
    pick: str = PICK_CALMAR,
) -> dict[str, float | str | bool | None]:
    """Select ``target_ann_vol`` on the research-IS surface only.

    ``max_oos_dd`` is kept as the keyword name used by notebooks. It is a positive
    fraction (0.25 = 25% hole) applied to ``is_max_drawdown``, which is stored
    negative. OOS columns are reported on the chosen row and are never the objective.
    """
    if pick not in VALID_PICK:
        raise ValueError(f"pick must be one of {sorted(VALID_PICK)}")
    if surface is None or surface.empty:
        return {
            "target_ann_vol": None,
            "pick": pick,
            "n_survivors": 0,
            "half_kelly_vol": float(half_kelly_vol),
            "max_oos_dd": float(max_oos_dd),
            "reason": "empty_surface",
        }
    df = surface.copy()
    cap = abs(float(max_oos_dd))
    hk = float(half_kelly_vol)
    df["below_half_kelly"] = df["target_ann_vol"].astype(float) <= hk + 1e-12
    df["dd_ok"] = df["is_max_drawdown"].astype(float) >= -cap
    survivors = df.loc[df["below_half_kelly"] & df["dd_ok"]].copy()
    col = "is_calmar" if pick == PICK_CALMAR else "is_cagr"
    if survivors.empty or survivors[col].notna().sum() == 0:
        return {
            "target_ann_vol": None,
            "pick": pick,
            "n_survivors": int(len(survivors)),
            "half_kelly_vol": hk,
            "max_oos_dd": cap,
            "reason": "no_survivor",
        }
    ranked = survivors.sort_values(col, ascending=False, na_position="last")
    best = ranked.iloc[0]

    def _f(name: str) -> float:
        if name not in best.index or pd.isna(best[name]):
            return float("nan")
        return float(best[name])

    return {
        "target_ann_vol": float(best["target_ann_vol"]),
        "pick": pick,
        "n_survivors": int(len(survivors)),
        "half_kelly_vol": hk,
        "max_oos_dd": cap,
        "is_calmar": _f("is_calmar"),
        "is_cagr": _f("is_cagr"),
        "is_max_drawdown": _f("is_max_drawdown"),
        "is_sharpe": _f("is_sharpe"),
        "oos_calmar": _f("oos_calmar"),
        "oos_cagr": _f("oos_cagr"),
        "oos_max_drawdown": _f("oos_max_drawdown"),
        "oos_sharpe": _f("oos_sharpe"),
        "reason": "ok",
    }


def k_fair_from_artifact(
    artifact: dict | None,
    sealed_returns: pd.Series,
    *,
    periods_per_year: float,
) -> float:
    """VT-equivalent scalar ``k_fair ≈ target_ann_vol / realized_sealed_vol``.

    Labeled starting gross only — not a second VT overlay. Missing artifact → 1.0.
    """
    if not artifact or artifact.get("target_ann_vol") is None:
        return 1.0
    r = pd.to_numeric(sealed_returns, errors="coerce").astype(float).dropna()
    if len(r) < 2:
        return 1.0
    vol = float(r.std(ddof=1) * np.sqrt(float(periods_per_year)))
    target = float(artifact["target_ann_vol"])
    if not np.isfinite(vol) or vol <= 1e-12 or not np.isfinite(target) or target <= 0:
        return 1.0
    return float(target / vol)
