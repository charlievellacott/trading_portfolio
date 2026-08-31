"""Stationary block bootstrap (univariate and joint)."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from risk.analytics.monte_carlo.simulator import MonteCarloSimulator


def stationary_bootstrap_indices(
    n_obs: int,
    horizon: int,
    n_simulations: int,
    mean_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary bootstrap index paths (Politis–Romano).

    Shape ``(horizon, n_simulations)``. Each column is one path of positions
    into ``[0, n_obs)``. Block restarts occur with probability
    ``1 / mean_block_length``; otherwise the next circular index is used.
    The same index path is applied to every asset (joint resampling).
    """
    if n_obs < 1:
        raise ValueError("n_obs must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")
    length = float(mean_block_length)
    if not np.isfinite(length) or length < 1.0:
        raise ValueError("mean_block_length must be finite and >= 1")
    p = min(1.0, 1.0 / length)
    idx = np.empty((horizon, n_simulations), dtype=np.intp)
    idx[0] = rng.integers(0, n_obs, size=n_simulations)
    if horizon == 1:
        return idx
    u = rng.random((horizon - 1, n_simulations))
    jumps = rng.integers(0, n_obs, size=(horizon - 1, n_simulations))
    for t in range(1, horizon):
        restart = u[t - 1] < p
        nxt = (idx[t - 1] + 1) % n_obs
        idx[t] = np.where(restart, jumps[t - 1], nxt)
    return idx


def is_joint_simulations(simulations: pd.DataFrame) -> bool:
    """True when columns are MultiIndex ``(simulation, asset)``."""
    return isinstance(simulations.columns, pd.MultiIndex)


def asset_paths(simulations: pd.DataFrame, asset: str | None = None) -> pd.DataFrame:
    """Extract a univariate path frame from joint or univariate simulations."""
    if not is_joint_simulations(simulations):
        return simulations
    level = "asset" if "asset" in (simulations.columns.names or []) else 1
    names = simulations.columns.get_level_values(level)
    if asset is None:
        if "strategy" in set(names):
            asset = "strategy"
        else:
            asset = str(names[0])
    return simulations.xs(asset, axis=1, level=level)


def split_joint_simulations(
    simulations: pd.DataFrame,
    *,
    strategy: str = "strategy",
    spy: str = "spy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split joint simulations into paired strategy and SPY path frames."""
    if not is_joint_simulations(simulations):
        raise ValueError("expected MultiIndex columns (simulation, asset)")
    return asset_paths(simulations, strategy), asset_paths(simulations, spy)


def _terminal_wealth_summary(paths: pd.DataFrame, label: str) -> pd.DataFrame:
    wealth = (1.0 + paths.astype(float)).prod(axis=0)
    qs = (0.05, 0.25, 0.50, 0.75, 0.95)
    row = {"asset": label, "n_paths": int(paths.shape[1]), "mean_wealth": float(wealth.mean())}
    for q in qs:
        row[f"p{int(q * 100):02d}_wealth"] = float(wealth.quantile(q))
    return pd.DataFrame([row]).set_index("asset")


class StationaryBlockBootstrap(MonteCarloSimulator):
    """Stationary block bootstrap of simple period returns.

    Fit on a ``Series`` (univariate paths for prop-firm storms) or a
    two-column ``DataFrame`` (same block indices on every column — required
    for ``P(beat SPY)``).
    """

    def __init__(
        self,
        n_simulations: int,
        horizon: int | None = None,
        random_seed: int | None = None,
        *,
        mean_block_length: float = 10.0,
    ) -> None:
        super().__init__(n_simulations, horizon, random_seed)
        self.mean_block_length = float(mean_block_length)
        self._values: np.ndarray | None = None
        self._names: list[str] = []
        self._index: pd.Index | None = None
        self._rng: np.random.Generator | None = None

    def fit(self, returns: pd.Series | pd.DataFrame) -> Self:
        if isinstance(returns, pd.Series):
            s = pd.to_numeric(returns, errors="coerce").astype(float).dropna()
            if s.empty:
                raise ValueError("returns series is empty after dropping NaNs")
            self._values = s.to_numpy(dtype=float).reshape(-1, 1)
            name = s.name if s.name is not None else "strategy"
            self._names = [str(name)]
            self._index = s.index
        elif isinstance(returns, pd.DataFrame):
            if returns.shape[1] < 1:
                raise ValueError("returns DataFrame has no columns")
            d = returns.apply(pd.to_numeric, errors="coerce").astype(float).dropna(how="any")
            if d.empty:
                raise ValueError("returns DataFrame is empty after dropping NaNs")
            self._values = d.to_numpy(dtype=float)
            self._names = [str(c) for c in d.columns]
            self._index = d.index
        else:
            raise TypeError("returns must be a Series or DataFrame")
        self._rng = np.random.default_rng(self.random_seed)
        return self

    def simulate(self, horizon: int) -> pd.DataFrame:
        if self._values is None or self._rng is None:
            raise RuntimeError("call fit() before simulate()")
        n_obs, n_assets = self._values.shape
        idx = stationary_bootstrap_indices(
            n_obs,
            int(horizon),
            int(self.n_simulations),
            self.mean_block_length,
            self._rng,
        )
        drawn = self._values[idx]  # (horizon, n_sim, n_assets)
        if n_assets == 1:
            data = drawn[:, :, 0]
            columns = [f"sim_{i}" for i in range(self.n_simulations)]
            return pd.DataFrame(data, columns=columns)
        arrays = []
        tuples: list[tuple[str, str]] = []
        for i in range(self.n_simulations):
            for k, name in enumerate(self._names):
                arrays.append(drawn[:, i, k])
                tuples.append((f"sim_{i}", name))
        columns = pd.MultiIndex.from_tuples(tuples, names=["simulation", "asset"])
        return pd.DataFrame(np.column_stack(arrays), columns=columns)

    def summary(self, simulations: pd.DataFrame) -> pd.DataFrame:
        if is_joint_simulations(simulations):
            level = "asset" if "asset" in (simulations.columns.names or []) else 1
            assets = list(dict.fromkeys(simulations.columns.get_level_values(level)))
            parts = [
                _terminal_wealth_summary(asset_paths(simulations, a), a) for a in assets
            ]
            return pd.concat(parts)
        return _terminal_wealth_summary(simulations, self._names[0] if self._names else "strategy")
