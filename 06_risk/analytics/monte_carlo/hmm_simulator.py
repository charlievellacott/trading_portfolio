"""Two-state Gaussian HMM path simulator (univariate strategy returns)."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from data.processing.feature_implementation.hmm_regime import (
    GaussianHMM2Params,
    fit_gaussian_hmm_2state,
)
from risk.analytics.monte_carlo.block_bootstrap import _terminal_wealth_summary
from risk.analytics.monte_carlo.simulator import MonteCarloSimulator


class GaussianHMMSimulator(MonteCarloSimulator):
    """Univariate 2-state Gaussian HMM. Not a substitute for joint SPY paths."""

    def __init__(
        self,
        n_simulations: int,
        horizon: int | None = None,
        random_seed: int | None = None,
        *,
        n_iter: int = 40,
    ) -> None:
        super().__init__(n_simulations, horizon, random_seed)
        self.n_iter = int(n_iter)
        self.params: GaussianHMM2Params | None = None
        self._rng: np.random.Generator | None = None
        self._name = "strategy"

    def fit(self, returns: pd.Series | pd.DataFrame) -> Self:
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] != 1:
                raise TypeError(
                    "GaussianHMMSimulator is univariate; pass a Series "
                    "(not a joint strategy/SPY frame)"
                )
            series = returns.iloc[:, 0]
        else:
            series = returns
        s = pd.to_numeric(series, errors="coerce").astype(float).dropna()
        if s.name is not None:
            self._name = str(s.name)
        em_seed = 0 if self.random_seed is None else int(self.random_seed)
        self.params = fit_gaussian_hmm_2state(s, n_iter=self.n_iter, seed=em_seed)
        self._rng = np.random.default_rng(self.random_seed)
        return self

    def simulate(self, horizon: int) -> pd.DataFrame:
        if self.params is None or self._rng is None:
            raise RuntimeError("call fit() before simulate()")
        h = int(horizon)
        n_sim = int(self.n_simulations)
        means = np.array(self.params.means, dtype=float)
        std = np.sqrt(np.maximum(np.array(self.params.vars, dtype=float), 1e-12))
        trans = np.array(self.params.trans, dtype=float)
        start = np.array(self.params.start, dtype=float)
        start = start / start.sum()
        states = self._rng.choice(2, size=n_sim, p=start)
        out = np.empty((h, n_sim), dtype=float)
        for t in range(h):
            out[t] = self._rng.normal(means[states], std[states])
            stay0 = trans[states, 0]
            u = self._rng.random(n_sim)
            states = np.where(u < stay0, 0, 1).astype(np.intp)
        columns = [f"sim_{i}" for i in range(n_sim)]
        return pd.DataFrame(out, columns=columns)

    def summary(self, simulations: pd.DataFrame) -> pd.DataFrame:
        return _terminal_wealth_summary(simulations, self._name)
