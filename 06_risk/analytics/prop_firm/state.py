"""Mutable phase state for a prop-firm evaluation path."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhaseState:
    """Account state after the last fully processed bar (EOD / closed)."""

    initial_capital: float
    balance: float
    equity: float
    sod_balance: float
    trading_days_opened: int = 0
    bar: int = -1
    status: str = "open"
    failure_reason: str | None = None
    first_binding: str | None = None
    target_hit_bar: int | None = None
    passed_bar: int | None = None
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    min_equity: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_equity <= 0.0:
            self.peak_equity = float(self.initial_capital)
        if self.min_equity <= 0.0:
            self.min_equity = float(self.initial_capital)


def initial_state(initial_capital: float) -> PhaseState:
    c = float(initial_capital)
    return PhaseState(
        initial_capital=c,
        balance=c,
        equity=c,
        sod_balance=c,
        peak_equity=c,
        min_equity=c,
    )


@dataclass(frozen=True)
class PathOutcome:
    """Immutable result of evaluating one return path against a phase."""

    passed: bool
    status: str
    failure_reason: str | None
    first_binding: str | None
    days_to_pass: int | None
    bars_elapsed: int
    final_balance: float
    final_return: float
    max_drawdown: float
    min_equity: float
    trading_days_opened: int
    target_hit_bar: int | None
    passed_bar: int | None
    dd_budget_at_pass: float | None
    min_days_delay: int | None = field(default=None)

    @classmethod
    def from_state(
        cls,
        state: PhaseState,
        *,
        max_loss_floor: float,
    ) -> PathOutcome:
        passed = state.status == "passed"
        delay = None
        if (
            state.passed_bar is not None
            and state.target_hit_bar is not None
        ):
            delay = int(state.passed_bar - state.target_hit_bar)
        budget = None
        if passed:
            budget = float(state.equity - max_loss_floor)
        days = None if state.passed_bar is None else int(state.passed_bar + 1)
        return cls(
            passed=passed,
            status=state.status,
            failure_reason=state.failure_reason,
            first_binding=state.first_binding,
            days_to_pass=days,
            bars_elapsed=int(state.bar + 1) if state.bar >= 0 else 0,
            final_balance=float(state.balance),
            final_return=float(state.balance / state.initial_capital - 1.0),
            max_drawdown=float(state.max_drawdown),
            min_equity=float(state.min_equity),
            trading_days_opened=int(state.trading_days_opened),
            target_hit_bar=state.target_hit_bar,
            passed_bar=state.passed_bar,
            dd_budget_at_pass=budget,
            min_days_delay=delay,
        )
