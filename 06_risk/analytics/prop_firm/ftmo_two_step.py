"""FTMO Challenge 2-Step phases (not 1-step)."""

from __future__ import annotations

import pandas as pd

from risk.monte_carlo.block_bootstrap import StationaryBlockBootstrap, asset_paths
from risk.monte_carlo.ev_stats import scale_simple_returns
from risk.prop_firm.base import ChallengePhase


class FtmoTwoStepChallenge(ChallengePhase):
    """FTMO Challenge phase: 10% closed-balance target, 4 min trading days."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("profit_target_frac", 0.10)
        kwargs.setdefault("min_trading_days", 4)
        kwargs.setdefault("max_daily_loss_frac", 0.05)
        kwargs.setdefault("max_loss_frac", 0.10)
        super().__init__(**kwargs)

    def registry_key(self) -> str:
        return "ftmo.2step.challenge"

    def name(self) -> str:
        return "FTMO 2-Step Challenge"


class FtmoTwoStepVerification(ChallengePhase):
    """FTMO Verification: 5% closed-balance target, 4 min trading days."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("profit_target_frac", 0.05)
        kwargs.setdefault("min_trading_days", 4)
        kwargs.setdefault("max_daily_loss_frac", 0.05)
        kwargs.setdefault("max_loss_frac", 0.10)
        super().__init__(**kwargs)

    def registry_key(self) -> str:
        return "ftmo.2step.verification"

    def name(self) -> str:
        return "FTMO 2-Step Verification"


class FtmoTwoStepFunded(ChallengePhase):
    """FTMO Account after 2-step: loss limits only (no target, no min days)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("profit_target_frac", None)
        kwargs.setdefault("min_trading_days", 0)
        kwargs.setdefault("max_daily_loss_frac", 0.05)
        kwargs.setdefault("max_loss_frac", 0.10)
        super().__init__(**kwargs)

    def registry_key(self) -> str:
        return "ftmo.2step.funded"

    def name(self) -> str:
        return "FTMO 2-Step Funded"


def run_two_step(
    challenge_paths: pd.DataFrame,
    verification_paths: pd.DataFrame,
    funded_paths: pd.DataFrame | None = None,
    *,
    initial_capital: float = 100_000.0,
    challenge: ChallengePhase | None = None,
    verification: ChallengePhase | None = None,
    funded: ChallengePhase | None = None,
) -> pd.DataFrame:
    """Evaluate paired independent paths: Challenge, then Verification, then funded.

    Each phase resets to ``initial_capital``. Columns must align by position.
    """
    chal = challenge or FtmoTwoStepChallenge()
    ver = verification or FtmoTwoStepVerification()
    fund = funded or FtmoTwoStepFunded()
    c_paths = asset_paths(challenge_paths)
    v_paths = asset_paths(verification_paths)
    if c_paths.shape[1] != v_paths.shape[1]:
        raise ValueError("challenge and verification path counts must match")
    c_res = chal.evaluate(c_paths, initial_capital=initial_capital)
    v_res = ver.evaluate(v_paths, initial_capital=initial_capital)
    c_res = c_res.add_prefix("chal_").reset_index(drop=True)
    v_res = v_res.add_prefix("verif_").reset_index(drop=True)
    out = pd.concat([c_res, v_res], axis=1)
    out["passed_challenge"] = out["chal_passed"].astype(bool)
    out["passed_verification"] = out["verif_passed"].astype(bool)
    out["passed_both"] = out["passed_challenge"] & out["passed_verification"]

    if funded_paths is not None:
        f_paths = asset_paths(funded_paths)
        if f_paths.shape[1] != c_paths.shape[1]:
            raise ValueError("funded path count must match challenge")
        f_res = fund.evaluate(f_paths, initial_capital=initial_capital)
        f_res = f_res.add_prefix("funded_").reset_index(drop=True)
        out = pd.concat([out, f_res], axis=1)
        survived = out["funded_status"].eq("survived")
        surplus = out["funded_final_balance"] - float(initial_capital)
        out["funded_surplus"] = surplus.where(out["passed_both"] & survived, 0.0)
    else:
        out["funded_surplus"] = 0.0
    out.index = pd.Index([f"sim_{i}" for i in range(len(out))], name="simulation")
    return out


def simulate_ftmo_two_step(
    period_returns: pd.Series,
    *,
    n_simulations: int,
    horizon: int,
    horizon_funded: int,
    mean_block_length: float = 10.0,
    leverage: float = 1.0,
    initial_capital: float = 100_000.0,
    random_seed: int = 0,
    always_in_market: bool = False,
) -> pd.DataFrame:
    """Univariate bootstrap storms through Challenge → Verification → Funded."""
    scaled = scale_simple_returns(period_returns, leverage)
    if not isinstance(scaled, pd.Series):
        scaled = scaled.iloc[:, 0]
    sim = StationaryBlockBootstrap(
        n_simulations,
        horizon=horizon,
        random_seed=random_seed,
        mean_block_length=mean_block_length,
    )
    sim.fit(scaled)
    chal_p = sim.simulate(int(horizon))
    ver_p = sim.simulate(int(horizon))
    fund_p = sim.simulate(int(horizon_funded))
    kw = dict(always_in_market=always_in_market)
    return run_two_step(
        chal_p,
        ver_p,
        fund_p,
        initial_capital=initial_capital,
        challenge=FtmoTwoStepChallenge(**kw),
        verification=FtmoTwoStepVerification(**kw),
        funded=FtmoTwoStepFunded(**kw),
    )
