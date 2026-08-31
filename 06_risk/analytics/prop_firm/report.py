"""Desk helpers for prop-firm notebooks (tables, leverage grid)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk.analytics.prop_firm.economics import attach_economics, geometric_attempts_until_pass
from risk.analytics.prop_firm.ftmo_two_step import simulate_ftmo_two_step


def two_step_headline(results: pd.DataFrame) -> pd.Series:
    n = len(results)
    if n == 0:
        return pd.Series(dtype=float, name="ftmo.2step")
    chal = results["passed_challenge"].astype(bool)
    ver = results["passed_verification"].astype(bool)
    both = results["passed_both"].astype(bool)
    chal_days = results.loc[chal, "chal_days_to_pass"]
    ver_days = results.loc[ver, "verif_days_to_pass"]
    both_days = None
    if both.any():
        both_days = (
            results.loc[both, "chal_days_to_pass"].astype(float)
            + results.loc[both, "verif_days_to_pass"].astype(float)
        )
    mix = results.loc[~chal, "chal_first_binding"].value_counts(dropna=False)
    out = {
        "p_challenge": float(chal.mean()),
        "p_verification": float(ver.mean()),
        "p_both": float(both.mean()),
        "median_days_challenge": float(chal_days.median()) if len(chal_days) else float("nan"),
        "p90_days_challenge": float(chal_days.quantile(0.90)) if len(chal_days) else float("nan"),
        "median_days_verification": float(ver_days.median()) if len(ver_days) else float("nan"),
        "median_days_both": float(both_days.median()) if both_days is not None else float("nan"),
        "p90_days_both": float(both_days.quantile(0.90)) if both_days is not None else float("nan"),
        "n_paths": float(n),
        "chal_fail_daily_loss": float(mix.get("daily_loss", 0)),
        "chal_fail_max_loss": float(mix.get("max_loss", 0)),
        "chal_incomplete": float((results["chal_status"] == "incomplete").mean()),
        "funded_survive_given_both": float(
            results.loc[both, "funded_status"].eq("survived").mean()
        )
        if both.any()
        else float("nan"),
    }
    return pd.Series(out, name="ftmo.2step")


def run_challenge_select(
    period_returns: pd.Series,
    *,
    n_simulations: int,
    horizon: int,
    horizon_funded: int,
    leverage: float,
    initial_capital: float,
    fee: float,
    profit_split: float,
    mean_block_length: float = 10.0,
    random_seed: int = 0,
    verification_fee: float = 0.0,
    always_in_market: bool = False,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Full two-step storm + economics for the selector notebook."""
    results = simulate_ftmo_two_step(
        period_returns,
        n_simulations=n_simulations,
        horizon=horizon,
        horizon_funded=horizon_funded,
        mean_block_length=mean_block_length,
        leverage=leverage,
        initial_capital=initial_capital,
        random_seed=random_seed,
        always_in_market=always_in_market,
    )
    cash, econ = attach_economics(
        results,
        fee=fee,
        profit_split=profit_split,
        verification_fee=verification_fee,
        horizon_days=float(horizon + horizon),
    )
    head = two_step_headline(results)
    combined = pd.concat([head, econ])
    retries = geometric_attempts_until_pass(
        float(head["p_both"]),
        n_simulations,
        random_seed=random_seed,
    )
    return {
        "results": results,
        "cashflows": cash,
        "economics": econ,
        "headline": combined,
        "retries_until_pass": pd.Series(retries, name="attempts"),
    }


def leverage_ev_grid(
    period_returns: pd.Series,
    leverages: list[float] | np.ndarray,
    *,
    n_simulations: int,
    horizon: int,
    horizon_funded: int,
    initial_capital: float,
    fee: float,
    profit_split: float,
    mean_block_length: float = 10.0,
    random_seed: int = 0,
    verification_fee: float = 0.0,
) -> pd.DataFrame:
    """Grid over ``k``: pass rates and EV/day (optimize EV, not pass rate)."""
    rows = []
    for k in leverages:
        pack = run_challenge_select(
            period_returns,
            n_simulations=n_simulations,
            horizon=horizon,
            horizon_funded=horizon_funded,
            leverage=float(k),
            initial_capital=initial_capital,
            fee=fee,
            profit_split=profit_split,
            mean_block_length=mean_block_length,
            random_seed=random_seed,
            verification_fee=verification_fee,
        )
        h = pack["headline"]
        res = pack["results"]
        failed = res["chal_status"].eq("failed")
        p_fail_daily = float((failed & res["chal_first_binding"].eq("daily_loss")).mean())
        p_fail_max = float((failed & res["chal_first_binding"].eq("max_loss")).mean())
        mix = res.loc[failed, "chal_first_binding"].value_counts(dropna=False)
        first_binding = str(mix.index[0]) if len(mix) else ""
        rows.append(
            {
                "leverage": float(k),
                "p_both": float(h["p_both"]),
                "p_challenge": float(h["p_challenge"]),
                "ev": float(h["ev"]),
                "ev_per_day": float(h["ev_per_day"]),
                "p_ev_le_0": float(h["p_ev_le_0"]),
                "do_not_take": bool(h["do_not_take"]),
                "ci_low": float(h["ci_low"]),
                "p_fail_daily_loss": p_fail_daily,
                "p_fail_max_loss": p_fail_max,
                "first_binding": first_binding,
            }
        )
    return pd.DataFrame(rows)


def suggest_challenge_leverage(
    grid: pd.DataFrame,
    *,
    k_fair: float,
    max_p_fail_daily: float = 0.40,
    max_p_fail_max: float = 0.30,
) -> dict[str, float | bool]:
    """``k_suggested = min(k_fair, k_ftmo)`` among rows inside fail-rate caps.

    ``k_ftmo`` maximises EV/day among leverages with
    ``p_fail_daily_loss`` / ``p_fail_max_loss`` at or below the caps and
    ``do_not_take`` false. Suggested k never exceeds the fail-rate cap set.
    """
    if grid is None or grid.empty:
        return {
            "k_fair": float(k_fair),
            "k_ftmo": float("nan"),
            "k_suggested": float("nan"),
            "do_not_take": True,
            "ev_per_day": float("nan"),
        }
    g = grid.copy()
    ok = (
        g["p_fail_daily_loss"].astype(float).le(float(max_p_fail_daily))
        & g["p_fail_max_loss"].astype(float).le(float(max_p_fail_max))
        & ~g["do_not_take"].astype(bool)
    )
    survivors = g.loc[ok]
    if survivors.empty:
        return {
            "k_fair": float(k_fair),
            "k_ftmo": float("nan"),
            "k_suggested": float("nan"),
            "do_not_take": True,
            "ev_per_day": float("nan"),
        }
    best = survivors.sort_values("ev_per_day", ascending=False).iloc[0]
    k_ftmo = float(best["leverage"])
    k_suggested = min(float(k_fair), k_ftmo)
    nearest_idx = (g["leverage"].astype(float) - k_suggested).abs().idxmin()
    near = g.loc[nearest_idx]
    return {
        "k_fair": float(k_fair),
        "k_ftmo": k_ftmo,
        "k_suggested": float(k_suggested),
        "do_not_take": bool(near.get("do_not_take", False)),
        "ev_per_day": float(best["ev_per_day"]),
        "p_fail_daily_loss": float(best["p_fail_daily_loss"]),
        "p_fail_max_loss": float(best["p_fail_max_loss"]),
        "p_both": float(best["p_both"]),
    }


def fair_vs_ftmo_row(
    grid: pd.DataFrame,
    suggestion: dict,
) -> pd.DataFrame:
    """Two-row compare: unconstrained fair VT vs FTMO-capped k."""
    def _nearest(k: float) -> pd.Series | None:
        if grid is None or grid.empty or not np.isfinite(k):
            return None
        idx = (grid["leverage"].astype(float) - float(k)).abs().idxmin()
        return grid.loc[idx]

    rows = []
    for label, key in (
        ("unconstrained fair VT", "k_fair"),
        ("FTMO-capped", "k_suggested"),
    ):
        k = float(suggestion.get(key, float("nan")))
        row = _nearest(k)
        rec = {"label": label, "k": k}
        if row is not None:
            rec.update(
                {
                    "ev_per_day": float(row["ev_per_day"]),
                    "p_both": float(row["p_both"]),
                    "p_fail_daily_loss": float(row["p_fail_daily_loss"]),
                    "p_fail_max_loss": float(row["p_fail_max_loss"]),
                    "do_not_take": bool(row["do_not_take"]),
                }
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def binding_mix(results: pd.DataFrame, *, phase: str = "chal") -> pd.Series:
    col = f"{phase}_first_binding"
    status = f"{phase}_status"
    failed = results.loc[results[status] == "failed", col]
    return failed.value_counts(dropna=False)
