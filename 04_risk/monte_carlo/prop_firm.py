"""Abstract prop-firm challenge evaluator for Monte Carlo return paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from risk.monte_carlo.simulator import MonteCarloSimulator

FAILURE_REASONS = ("daily_loss", "max_drawdown", "timeout")


@dataclass(frozen=True)
class PropFirmRules:
    """Thresholds for a prop-firm challenge evaluation."""

    profit_target: float
    max_drawdown: float
    daily_loss_limit: float
    window_days: int


@dataclass(frozen=True)
class PathResult:
    """Outcome of evaluating a single simulated return path."""

    passed: bool
    failure_reason: str | None
    days_to_pass: int | None
    final_return: float
    max_drawdown: float


class PropFirmChecker(ABC):
    """Evaluate simulated return paths against prop-firm challenge rules.

    Consumes the ``DataFrame`` produced by :meth:`MonteCarloSimulator.simulate`:

    - **Rows** = trading days (``0 .. horizon-1``)
    - **Columns** = independent paths (``sim_0``, ``sim_1``, … or integer names)
    - **Values** = simple period returns

    Per-bar check order: daily loss → max drawdown → profit target.
    First breach ends the path. If the window ends without breach or target,
    the path fails with ``failure_reason="timeout"``.

    Documented ``failure_reason`` values: ``"daily_loss"``, ``"max_drawdown"``,
    ``"timeout"``. Passed paths set ``failure_reason`` to ``None``.
  """

    @abstractmethod
    def rules(self) -> PropFirmRules:
        """Return firm-specific challenge thresholds."""

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable firm label for summaries."""

    def evaluate_path(
        self,
        returns: pd.Series,
        *,
        initial_balance: float = 1.0,
    ) -> PathResult:
        """Evaluate one return path against :meth:`rules`."""
        rules = self.rules()
        window_returns = returns.iloc[: rules.window_days]

        if window_returns.empty:
            return PathResult(
                passed=False,
                failure_reason="timeout",
                days_to_pass=None,
                final_return=0.0,
                max_drawdown=0.0,
            )

        equity = initial_balance * (1.0 + window_returns).cumprod()
        peak = initial_balance
        worst_dd = 0.0

        for day, (period_return, eq) in enumerate(zip(window_returns, equity, strict=True)):
            if period_return < -rules.daily_loss_limit:
                return PathResult(
                    passed=False,
                    failure_reason="daily_loss",
                    days_to_pass=None,
                    final_return=float(eq / initial_balance - 1.0),
                    max_drawdown=worst_dd,
                )

            if eq > peak:
                peak = eq
            drawdown = eq / peak - 1.0
            if drawdown < worst_dd:
                worst_dd = drawdown
            if drawdown < -rules.max_drawdown:
                return PathResult(
                    passed=False,
                    failure_reason="max_drawdown",
                    days_to_pass=None,
                    final_return=float(eq / initial_balance - 1.0),
                    max_drawdown=worst_dd,
                )

            total_return = eq / initial_balance - 1.0
            if total_return >= rules.profit_target:
                return PathResult(
                    passed=True,
                    failure_reason=None,
                    days_to_pass=day,
                    final_return=float(total_return),
                    max_drawdown=worst_dd,
                )

        final_return = float(equity.iloc[-1] / initial_balance - 1.0)
        return PathResult(
            passed=False,
            failure_reason="timeout",
            days_to_pass=None,
            final_return=final_return,
            max_drawdown=worst_dd,
        )

    def evaluate(
        self,
        simulations: pd.DataFrame,
        *,
        initial_balance: float = 1.0,
    ) -> pd.DataFrame:
        """Evaluate each simulated path (one column) against :meth:`rules`."""
        rows = []
        for column in simulations.columns:
            result = self.evaluate_path(
                simulations[column],
                initial_balance=initial_balance,
            )
            rows.append(
                {
                    "simulation": column,
                    "passed": result.passed,
                    "failure_reason": result.failure_reason,
                    "days_to_pass": result.days_to_pass,
                    "final_return": result.final_return,
                    "max_drawdown": result.max_drawdown,
                }
            )

        return pd.DataFrame(rows).set_index("simulation")

    def summary(self, results: pd.DataFrame) -> pd.Series:
        """Aggregate per-path results into challenge-level metrics."""
        metrics: dict[str, float] = {
            "pass_rate": float(results["passed"].mean()),
        }

        passes = results.loc[results["passed"]]
        if passes.empty:
            metrics["median_days_to_pass"] = float("nan")
        else:
            metrics["median_days_to_pass"] = float(passes["days_to_pass"].median())

        for reason in FAILURE_REASONS:
            metrics[f"failure_{reason}"] = float(
                (results["failure_reason"] == reason).sum()
            )

        return pd.Series(metrics, name=self.name())

    # this function when called will 
    # 1. fit the simulator to the returns
    # 2. generate paths
    # 3. evaluate the paths
    # 4. return the results and the summary
    # thus making it the key function to call to run the monte carlo simulation and evaluation.
    # you could call everything manually, but this function is a convenience function to call everything in one go.
    def run(
        self,
        simulator: MonteCarloSimulator, # any subclass of the simulator can be used here
        returns: pd.Series,
        *,
        horizon: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit a simulator, generate paths, and evaluate them."""
        simulator.fit(returns)
        resolved_horizon = horizon or simulator.horizon or self.rules().window_days
        if resolved_horizon is None:
            raise ValueError(
                "horizon must be provided via argument, simulator.horizon, "
                "or rules().window_days"
            )

        simulations = simulator.simulate(resolved_horizon)
        results = self.evaluate(simulations)
        return results, self.summary(results)
