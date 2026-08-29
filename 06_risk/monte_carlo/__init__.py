"""Monte Carlo edge-quality: EV, significance, joint P(beat SPY)."""

from risk.monte_carlo.block_bootstrap import (
    StationaryBlockBootstrap,
    asset_paths,
    is_joint_simulations,
    split_joint_simulations,
    stationary_bootstrap_indices,
)
from risk.monte_carlo.ev_stats import (
    apply_cost_haircut,
    cvar,
    ev_significance,
    excess_returns,
    hac_mean_inference,
    horizon_ev,
    p_not_beat_spy,
    p_not_beat_spy_from_joint,
    scale_simple_returns,
    simulate_joint_paths,
    underwater_probs,
)
from risk.monte_carlo.hmm_simulator import GaussianHMMSimulator
from risk.monte_carlo.simulator import MonteCarloSimulator

__all__ = [
    "GaussianHMMSimulator",
    "MonteCarloSimulator",
    "StationaryBlockBootstrap",
    "apply_cost_haircut",
    "asset_paths",
    "cvar",
    "ev_significance",
    "excess_returns",
    "hac_mean_inference",
    "horizon_ev",
    "is_joint_simulations",
    "p_not_beat_spy",
    "p_not_beat_spy_from_joint",
    "scale_simple_returns",
    "simulate_joint_paths",
    "split_joint_simulations",
    "stationary_bootstrap_indices",
    "underwater_probs",
]
