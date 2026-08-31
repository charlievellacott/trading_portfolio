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
    """Select ``target_ann_vol`` among rows that pass veto and sit at/below half-Kelly.

    ``max_oos_dd`` is a positive fraction (0.25 = 25% hole). ``oos_max_drawdown``
    is stored negative. A breaching row is never selected.
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
    df["dd_ok"] = df["oos_max_drawdown"].astype(float) >= -cap
    survivors = df.loc[df["below_half_kelly"] & df["dd_ok"]].copy()
    col = "oos_calmar" if pick == PICK_CALMAR else "oos_cagr"
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
    return {
        "target_ann_vol": float(best["target_ann_vol"]),
        "pick": pick,
        "n_survivors": int(len(survivors)),
        "half_kelly_vol": hk,
        "max_oos_dd": cap,
        "oos_calmar": float(best["oos_calmar"]) if pd.notna(best["oos_calmar"]) else float("nan"),
        "oos_cagr": float(best["oos_cagr"]) if pd.notna(best["oos_cagr"]) else float("nan"),
        "oos_max_drawdown": float(best["oos_max_drawdown"]),
        "oos_sharpe": float(best["oos_sharpe"]) if pd.notna(best["oos_sharpe"]) else float("nan"),
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
