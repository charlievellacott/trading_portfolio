"""Registry so later agents add firms without rewriting notebooks."""

from __future__ import annotations

from risk.prop_firm.base import ChallengePhase
from risk.prop_firm.ftmo_two_step import (
    FtmoTwoStepChallenge,
    FtmoTwoStepFunded,
    FtmoTwoStepVerification,
)

CHALLENGES: dict[str, type[ChallengePhase]] = {
    "ftmo.2step.challenge": FtmoTwoStepChallenge,
    "ftmo.2step.verification": FtmoTwoStepVerification,
    "ftmo.2step.funded": FtmoTwoStepFunded,
}


def make_challenge(key: str, **kwargs) -> ChallengePhase:
    if key not in CHALLENGES:
        known = ", ".join(sorted(CHALLENGES))
        raise KeyError(f"unknown challenge {key!r}; registered: {known}")
    return CHALLENGES[key](**kwargs)


def register_challenge(key: str, cls: type[ChallengePhase]) -> None:
    """Add or replace a challenge class (for future firms)."""
    CHALLENGES[key] = cls
