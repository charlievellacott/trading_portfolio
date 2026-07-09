"""Abstract Monte Carlo simulator for return-path generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import pandas as pd


class MonteCarloSimulator(ABC):
    """Abstract interface for Monte Carlo path generators.

    Bootstrap note: a bootstrap simulator creates many paths by resampling
    historical returns; block bootstrap resamples consecutive chunks.
    """

    def __init__(
        self,
        n_simulations: int,
        horizon: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.random_seed = random_seed

    @abstractmethod
    def fit(self, returns: pd.Series) -> Self:
        """Fit the simulator on historical period returns."""

    @abstractmethod
    def simulate(self, horizon: int) -> pd.DataFrame:
        """Generate simulated return paths.

        Returns a DataFrame with rows = trading days and columns = paths.
        """

    @abstractmethod
    def summary(self, simulations: pd.DataFrame) -> pd.DataFrame:
        """Summarize simulated paths (e.g. percentiles across simulations)."""
