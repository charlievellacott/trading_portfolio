"""Abstract prop-firm challenge phase (bar-by-bar EOD equity)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from risk.prop_firm.state import PathOutcome, PhaseState, initial_state


def first_binding_reason(
    equity: float,
    daily_limit: float,
    max_floor: float,
) -> str | None:
    """Constraint hit first on a declining path = the higher (tighter) floor."""
    hit_daily = equity < daily_limit
    hit_max = equity < max_floor
    if not hit_daily and not hit_max:
        return None
    if hit_daily and hit_max:
        return "daily_loss" if daily_limit >= max_floor else "max_loss"
    if hit_daily:
        return "daily_loss"
    return "max_loss"


class ChallengePhase(ABC):
    """One evaluation phase: profit target, loss limits, min trading days.

    Equity proxy is EOD marked wealth with positions treated as closed at the
    bar close (no intra-day high/low, no open PnL, no swaps). A bar counts as
    a trading day if ``|r| > open_eps`` unless ``always_in_market`` is True.

    Outcomes: ``passed``, ``failed`` (``daily_loss`` / ``max_loss``),
    ``incomplete`` (horizon ended without pass or fail), ``survived``
    (funded: no target, no breach through the horizon). ``incomplete`` is not
    an FTMO rule violation.
    """

    def __init__(
        self,
        *,
        profit_target_frac: float | None,
        min_trading_days: int,
        max_daily_loss_frac: float,
        max_loss_frac: float,
        open_eps: float = 1e-12,
        always_in_market: bool = False,
    ) -> None:
        self._profit_target_frac = profit_target_frac
        self._min_trading_days = int(min_trading_days)
        self._max_daily_loss_frac = float(max_daily_loss_frac)
        self._max_loss_frac = float(max_loss_frac)
        self.open_eps = float(open_eps)
        self.always_in_market = bool(always_in_market)

    @abstractmethod
    def registry_key(self) -> str:
        """Stable key for ``CHALLENGES`` (e.g. ``ftmo.2step.challenge``)."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable label."""

    def profit_target_frac(self) -> float | None:
        return self._profit_target_frac

    def min_trading_days(self) -> int:
        return self._min_trading_days

    def max_daily_loss_frac(self) -> float:
        return self._max_daily_loss_frac

    def max_loss_frac(self) -> float:
        return self._max_loss_frac

    def max_loss_floor(self, initial_capital: float) -> float:
        return (1.0 - self.max_loss_frac()) * float(initial_capital)

    def daily_loss_amount(self, initial_capital: float) -> float:
        return self.max_daily_loss_frac() * float(initial_capital)

    def daily_limit(self, sod_balance: float, initial_capital: float) -> float:
        return float(sod_balance) - self.daily_loss_amount(initial_capital)

    def bar_opened(self, period_return: float) -> bool:
        if self.always_in_market:
            return True
        return abs(float(period_return)) > self.open_eps

    def step(self, state: PhaseState, period_return: float) -> PhaseState:
        """Advance one EOD bar. No-op if the path already stopped."""
        if state.status != "open":
            return state
        r = float(period_return)
        cap = state.initial_capital
        state.bar += 1
        equity = state.balance * (1.0 + r)
        state.equity = equity
        if equity > state.peak_equity:
            state.peak_equity = equity
        if equity < state.min_equity:
            state.min_equity = equity
        dd = equity / state.peak_equity - 1.0 if state.peak_equity > 0 else 0.0
        if dd < state.max_drawdown:
            state.max_drawdown = dd

        daily_lim = self.daily_limit(state.sod_balance, cap)
        max_floor = self.max_loss_floor(cap)
        reason = first_binding_reason(equity, daily_lim, max_floor)
        if reason is not None:
            state.status = "failed"
            state.failure_reason = reason
            state.first_binding = reason
            state.balance = equity
            return state

        state.balance = equity
        if self.bar_opened(r):
            state.trading_days_opened += 1

        target = self.profit_target_frac()
        if target is not None and state.balance >= cap * (1.0 + float(target)):
            if state.target_hit_bar is None:
                state.target_hit_bar = state.bar
            if state.trading_days_opened >= self.min_trading_days():
                state.status = "passed"
                state.passed_bar = state.bar
                return state

        # Midnight CE(S)T proxy: next bar's SOD balance is this close.
        state.sod_balance = state.balance
        return state

    def finalize(self, state: PhaseState) -> PhaseState:
        if state.status != "open":
            return state
        if self.profit_target_frac() is None:
            state.status = "survived"
        else:
            state.status = "incomplete"
        return state

    def evaluate_path(
        self,
        returns: pd.Series,
        *,
        initial_capital: float = 100_000.0,
    ) -> PathOutcome:
        state = initial_state(initial_capital)
        for r in returns.astype(float).to_numpy():
            state = self.step(state, float(r))
            if state.status != "open":
                break
        state = self.finalize(state)
        return PathOutcome.from_state(
            state, max_loss_floor=self.max_loss_floor(initial_capital)
        )

    def evaluate(
        self,
        simulations: pd.DataFrame,
        *,
        initial_capital: float = 100_000.0,
    ) -> pd.DataFrame:
        rows = []
        for column in simulations.columns:
            out = self.evaluate_path(
                simulations[column],
                initial_capital=initial_capital,
            )
            rows.append(
                {
                    "simulation": column,
                    "passed": out.passed,
                    "status": out.status,
                    "failure_reason": out.failure_reason,
                    "first_binding": out.first_binding,
                    "days_to_pass": out.days_to_pass,
                    "bars_elapsed": out.bars_elapsed,
                    "final_balance": out.final_balance,
                    "final_return": out.final_return,
                    "max_drawdown": out.max_drawdown,
                    "min_equity": out.min_equity,
                    "trading_days_opened": out.trading_days_opened,
                    "target_hit_bar": out.target_hit_bar,
                    "passed_bar": out.passed_bar,
                    "dd_budget_at_pass": out.dd_budget_at_pass,
                    "min_days_delay": out.min_days_delay,
                }
            )
        return pd.DataFrame(rows).set_index("simulation")

    def summary(self, results: pd.DataFrame) -> pd.Series:
        n = float(len(results)) if len(results) else float("nan")
        passed = results["passed"] if "passed" in results.columns else pd.Series(dtype=bool)
        metrics: dict[str, float] = {
            "n_paths": float(len(results)),
            "pass_rate": float(passed.mean()) if n == n and len(results) else float("nan"),
            "fail_rate": float((results["status"] == "failed").mean()) if len(results) else float("nan"),
            "incomplete_rate": float((results["status"] == "incomplete").mean())
            if len(results)
            else float("nan"),
            "survived_rate": float((results["status"] == "survived").mean())
            if len(results)
            else float("nan"),
        }
        wins = results.loc[results["passed"]] if "passed" in results.columns else results.iloc[0:0]
        if wins.empty:
            metrics["median_days_to_pass"] = float("nan")
            metrics["p90_days_to_pass"] = float("nan")
        else:
            metrics["median_days_to_pass"] = float(wins["days_to_pass"].median())
            metrics["p90_days_to_pass"] = float(wins["days_to_pass"].quantile(0.90))
        for reason in ("daily_loss", "max_loss"):
            metrics[f"failure_{reason}"] = float(
                (results["first_binding"] == reason).sum()
            ) if len(results) else 0.0
        return pd.Series(metrics, name=self.name())
