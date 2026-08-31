"""Desk report for EV vs SPY notebooks."""

from __future__ import annotations

import pandas as pd

from risk.analytics.monte_carlo.ev_stats import (
    cvar,
    ev_significance,
    excess_returns,
    horizon_ev,
    p_not_beat_spy,
    scale_simple_returns,
    simulate_joint_paths,
    split_joint_simulations,
    terminal_simple_return,
    underwater_probs,
)
from risk.analytics.monte_carlo.geometry import (
    ev_concentration,
    excess_wealth_paths,
    holes_summary,
    joint_shape_vs_spy,
    pathwise_holes,
    realized_first_window_wealth,
    realized_terminal_percentile,
    realized_wealth_for_fan,
)
from risk.analytics.monte_carlo.plots import (
    equity_fan_figure,
    excess_wealth_fan_figure,
    max_dd_hist_figure,
    significance_frame,
    terminal_hist_figure,
    terminal_vs_dd_scatter_figure,
)


def run_ev_vs_spy(
    frame: pd.DataFrame,
    *,
    n_simulations: int,
    horizon: int,
    leverage: float,
    mean_block_length: float,
    periods_per_year: float,
    random_seed: int = 0,
    n_bootstrap: int = 1000,
    haircut_bps: float = 0.0,
) -> dict:
    """Historical EV significance + joint MC vs SPY (no prop-firm metrics)."""
    work = frame.copy()
    if haircut_bps:
        work["strategy"] = work["strategy"].astype(float) - float(haircut_bps) / 1.0e4
    scaled = scale_simple_returns(work, leverage)
    if not isinstance(scaled, pd.DataFrame):
        raise TypeError("frame must include strategy and spy columns")
    strat_hist = scaled["strategy"]
    spy_hist = scaled["spy"]
    hist_sig = ev_significance(
        strat_hist,
        periods_per_year=periods_per_year,
        mean_block_length=mean_block_length,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    excess = excess_returns(strat_hist, spy_hist)
    excess_sig = ev_significance(
        excess,
        periods_per_year=periods_per_year,
        mean_block_length=mean_block_length,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    joint = simulate_joint_paths(
        scaled,
        n_simulations=n_simulations,
        horizon=horizon,
        mean_block_length=mean_block_length,
        random_seed=random_seed,
        leverage=1.0,  # already scaled
    )
    s_paths, b_paths = split_joint_simulations(joint)
    hev = horizon_ev(s_paths)
    p_lose = p_not_beat_spy(s_paths, b_paths)
    uw = underwater_probs(s_paths)
    term = terminal_simple_return(s_paths)
    holes = pathwise_holes(s_paths)
    hole_sum = holes_summary(holes)
    conc = ev_concentration(s_paths)
    joint_shape = joint_shape_vs_spy(s_paths, b_paths)
    xs_w = excess_wealth_paths(s_paths, b_paths)

    r_s = realized_wealth_for_fan(strat_hist, horizon)
    r_b = realized_wealth_for_fan(spy_hist, horizon)
    r_s0 = realized_first_window_wealth(strat_hist, horizon)
    r_b0 = realized_first_window_wealth(spy_hist, horizon)
    r_xs = None
    if len(r_s) and len(r_b) and len(r_s) == len(r_b):
        r_xs = (r_s / r_b.replace(0.0, float("nan"))).rename("realized_excess")
    oos_pct = realized_terminal_percentile(r_s, s_paths)

    headline = pd.Series(
        {
            "hist_mean": hist_sig["mean"],
            "hist_t_stat": hist_sig["t_stat"],
            "hist_p_value": hist_sig["p_value"],
            "hist_ci_excludes_zero": hist_sig["ci_excludes_zero"],
            "hist_bootstrap_p_mean_le_0": hist_sig["bootstrap_p_mean_le_0"],
            "excess_mean": excess_sig["mean"],
            "excess_t_stat": excess_sig["t_stat"],
            "excess_p_value": excess_sig["p_value"],
            "excess_ci_excludes_zero": excess_sig["ci_excludes_zero"],
            "horizon_mean_terminal": hev["mean_terminal"],
            "horizon_median_terminal": conc["median_terminal"],
            "horizon_ci_low": hev["ci_low"],
            "horizon_ci_high": hev["ci_high"],
            "p_not_beat_spy": p_lose,
            "cvar_5": cvar(term, alpha=0.05),
            "max_dd_median": hole_sum["max_dd_median"],
            "max_dd_p05": hole_sum["max_dd_p05"],
            "top_decile_ev_share": conc["top_decile_ev_share"],
            "beta_median": joint_shape["beta_median"],
            "corr_median": joint_shape["corr_median"],
            "down_capture_median": joint_shape["down_capture_median"],
            "p_not_beat_given_spy_underwater": joint_shape[
                "p_not_beat_given_spy_underwater"
            ],
            "oos_terminal_percentile": oos_pct,
            "p_terminal_underwater": uw["p_terminal_underwater"],
            "p_ever_underwater": uw["p_ever_underwater"],
            "leverage": float(leverage),
            "horizon": float(horizon),
            "n_simulations": float(n_simulations),
        },
        name="ev_vs_spy",
    )
    return {
        "headline": headline,
        "hist_sig": hist_sig,
        "excess_sig": excess_sig,
        "hist_table": significance_frame(hist_sig),
        "excess_table": significance_frame(excess_sig),
        "strategy_paths": s_paths,
        "spy_paths": b_paths,
        "holes": holes,
        "holes_summary": hole_sum,
        "concentration": conc,
        "joint_shape": joint_shape,
        "fan": equity_fan_figure(
            s_paths,
            b_paths,
            realized_strategy=r_s,
            realized_spy=r_b,
            realized_strategy_first=r_s0,
            realized_spy_first=r_b0,
        ),
        "excess_fan": excess_wealth_fan_figure(xs_w, realized_excess=r_xs),
        "max_dd_hist": max_dd_hist_figure(holes["max_dd"]),
        "dd_scatter": terminal_vs_dd_scatter_figure(holes),
        "terminals": terminal_hist_figure(s_paths, b_paths),
    }
