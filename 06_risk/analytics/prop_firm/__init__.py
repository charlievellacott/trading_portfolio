"""Modular prop-firm challenge engine (separate from EV vs SPY)."""

from risk.prop_firm.base import ChallengePhase
from risk.prop_firm.economics import attach_economics, economic_ev_summary, pay_once_cashflows
from risk.prop_firm.ftmo_two_step import (
    FtmoTwoStepChallenge,
    FtmoTwoStepFunded,
    FtmoTwoStepVerification,
    run_two_step,
    simulate_ftmo_two_step,
)
from risk.prop_firm.registry import CHALLENGES, make_challenge, register_challenge
from risk.prop_firm.s1_calendar import weekly_to_weekday_returns
from risk.prop_firm.state import PathOutcome, PhaseState

__all__ = [
    "CHALLENGES",
    "ChallengePhase",
    "FtmoTwoStepChallenge",
    "FtmoTwoStepFunded",
    "FtmoTwoStepVerification",
    "PathOutcome",
    "PhaseState",
    "attach_economics",
    "economic_ev_summary",
    "make_challenge",
    "pay_once_cashflows",
    "register_challenge",
    "run_two_step",
    "simulate_ftmo_two_step",
    "weekly_to_weekday_returns",
]
