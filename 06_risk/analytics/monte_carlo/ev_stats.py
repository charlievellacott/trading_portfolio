"""Expected-value statistics and significance for sealed period returns."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from performance.sharpe_inference import (
    probabilistic_sharpe_ratio,
    return_moments,
)
from risk.analytics.monte_carlo.block_bootstrap import (
    StationaryBlockBootstrap,
    asset_paths,
    is_joint_simulations,
    split_joint_simulations,
    stationary_bootstrap_indices,
)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _two_sided_normal_p(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return float("nan")
    return float(2.0 * (1.0 - _norm_cdf(abs(float(t_stat)))))


def newey_west_lag(n: int) -> int:
    """Automatic Newey–West lag: ``floor(4 * (n/100)^{2/9})``."""
    if n < 2:
        return 0
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_mean_se(values: np.ndarray, *, lags: int) -> float:
    """HAC standard error of the sample mean (Bartlett / Newey–West)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return float("nan")
    mu = float(x.mean())
    e = x - mu
    gamma0 = float(np.dot(e, e) / n)
    long_run = gamma0
    l = max(0, int(lags))
    for j in range(1, l + 1):
        w = 1.0 - j / (l + 1.0)
        gamma = float(np.dot(e[j:], e[:-j]) / n)
        long_run += 2.0 * w * gamma
    if long_run <= 0.0:
        iid = float(np.var(x, ddof=1) / n)
        if iid <= 0.0:
            return 0.0
        return math.sqrt(iid)
    return float(math.sqrt(long_run / n))


def _clean_returns(returns: pd.Series) -> pd.Series:
    s = pd.to_numeric(returns, errors="coerce").astype(float).dropna()
    return s.sort_index() if isinstance(s.index, pd.DatetimeIndex) else s


def hac_mean_inference(
    returns: pd.Series,
    *,
    lags: int | None = None,
    level: float = 0.95,
    periods_per_year: float = 252.0,
) -> dict[str, float | bool | int]:
    """Newey–West inference for ``H0: E[r] = 0``.

    Headline fields: ``t_stat``, ``p_value``, ``ci_excludes_zero``.
    This is significance of the *historical mean*, not of a simulated path.
    """
    empty = {
        "mean": float("nan"),
        "mean_ann": float("nan"),
        "n_obs": 0,
        "periods_per_year": float(periods_per_year),
        "hac_lags": 0,
        "hac_se": float("nan"),
        "t_stat": float("nan"),
        "p_value": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "ci_excludes_zero": False,
        "level": float(level),
    }
    s = _clean_returns(returns)
    n = int(s.shape[0])
    if n < 2:
        return {**empty, "n_obs": n}
    mu = float(s.mean())
    used_lags = int(newey_west_lag(n) if lags is None else lags)
    se = newey_west_mean_se(s.to_numpy(), lags=used_lags)
    if not np.isfinite(se):
        t_stat = float("nan")
        p_value = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
        excludes = False
    elif se == 0.0:
        t_stat = (
            float("inf") if mu > 0.0 else float("-inf") if mu < 0.0 else 0.0
        )
        p_value = 0.0 if mu != 0.0 else 1.0
        ci_low = mu
        ci_high = mu
        excludes = mu != 0.0
    else:
        t_stat = float(mu / se)
        p_value = _two_sided_normal_p(t_stat)
        z = 1.959963984540054  # Phi^{-1}(0.975); used when level==0.95
        if abs(float(level) - 0.95) > 1e-12:
            target = 0.5 + float(level) / 2.0
            z = 0.0
            for _ in range(20):
                err = _norm_cdf(z) - target
                pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
                if pdf <= 0.0:
                    break
                z -= err / pdf
        half = z * se
        ci_low = mu - half
        ci_high = mu + half
        excludes = bool(ci_high < 0.0 or ci_low > 0.0)
    return {
        "mean": mu,
        "mean_ann": float(mu * periods_per_year),
        "n_obs": n,
        "periods_per_year": float(periods_per_year),
        "hac_lags": used_lags,
        "hac_se": float(se),
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": float(ci_low) if np.isfinite(ci_low) else float("nan"),
        "ci_high": float(ci_high) if np.isfinite(ci_high) else float("nan"),
        "ci_excludes_zero": excludes,
        "level": float(level),
    }


def bootstrap_mean_distribution(
    returns: pd.Series,
    *,
    mean_block_length: float = 10.0,
    n_bootstrap: int = 2000,
    random_seed: int | None = 0,
) -> np.ndarray:
    """Stationary-bootstrap replicates of the sample mean (length = n_obs)."""
    s = _clean_returns(returns)
    x = s.to_numpy(dtype=float)
    n = int(x.size)
    if n < 1 or n_bootstrap < 1:
        return np.array([], dtype=float)
    rng = np.random.default_rng(random_seed)
    idx = stationary_bootstrap_indices(
        n, n, int(n_bootstrap), float(mean_block_length), rng
    )
    return x[idx].mean(axis=0)


def bootstrap_mean_inference(
    returns: pd.Series,
    *,
    mean_block_length: float = 10.0,
    n_bootstrap: int = 2000,
    random_seed: int | None = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Percentile CI for the mean and ``P*(mu* <= 0)``."""
    means = bootstrap_mean_distribution(
        returns,
        mean_block_length=mean_block_length,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    if means.size == 0:
        return {
            "bootstrap_p_mean_le_0": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "n_bootstrap": 0,
        }
    alpha = (1.0 - float(level)) / 2.0
    return {
        "bootstrap_p_mean_le_0": float((means <= 0.0).mean()),
        "bootstrap_ci_low": float(np.quantile(means, alpha)),
        "bootstrap_ci_high": float(np.quantile(means, 1.0 - alpha)),
        "n_bootstrap": int(means.size),
    }


def ev_significance(
    returns: pd.Series,
    *,
    periods_per_year: float = 252.0,
    hac_lags: int | None = None,
    mean_block_length: float = 10.0,
    n_bootstrap: int = 2000,
    random_seed: int | None = 0,
    level: float = 0.95,
    sr_benchmark: float = 0.0,
) -> pd.Series:
    """EV point estimate plus the required significance bundle.

    Historical mean, HAC t/p/CI (``ci_excludes_zero``), bootstrap
    ``P*(mu* <= 0)``, and PSR as a *secondary* Sharpe-quality number.
    """
    hac = hac_mean_inference(
        returns,
        lags=hac_lags,
        level=level,
        periods_per_year=periods_per_year,
    )
    boot = bootstrap_mean_inference(
        returns,
        mean_block_length=mean_block_length,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        level=level,
    )
    s = _clean_returns(returns)
    moments = return_moments(s, periods_per_year=periods_per_year)
    psr = probabilistic_sharpe_ratio(
        moments["sr"],
        moments["n_obs"],
        moments["skew"],
        moments["kurtosis"],
        sr_benchmark=sr_benchmark,
    )
    psr1 = probabilistic_sharpe_ratio(
        moments["sr"],
        moments["n_obs"],
        moments["skew"],
        moments["kurtosis"],
        sr_benchmark=1.0,
    )
    out = {
        **hac,
        **boot,
        "sr": moments["sr"],
        "psr": float(psr) if psr is not None else float("nan"),
        "psr_vs_1": float(psr1) if psr1 is not None else float("nan"),
        "sr_benchmark": float(sr_benchmark),
    }
    return pd.Series(out, name=s.name if s.name is not None else "returns")


def excess_returns(strategy: pd.Series, spy: pd.Series) -> pd.Series:
    """Aligned ``r_strategy - r_spy`` (inner join on the index)."""
    a = _clean_returns(strategy).rename("strategy")
    b = _clean_returns(spy).rename("spy")
    j = pd.concat([a, b], axis=1).dropna(how="any")
    if j.empty:
        return pd.Series(dtype=float, name="excess")
    out = j["strategy"] - j["spy"]
    out.name = "excess"
    return out


def scale_simple_returns(
    returns: pd.Series | pd.DataFrame,
    leverage: float,
    *,
    strategy_col: str = "strategy",
) -> pd.Series | pd.DataFrame:
    """Risk-budget overlay ``r' = k r`` on the strategy (not a VT re-run).

    SPY columns are left unchanged when ``returns`` is a DataFrame.
    """
    k = float(leverage)
    if isinstance(returns, pd.Series):
        return returns.astype(float) * k
    d = returns.copy()
    if strategy_col in d.columns:
        d[strategy_col] = d[strategy_col].astype(float) * k
    else:
        first = d.columns[0]
        d[first] = d[first].astype(float) * k
    return d


def apply_cost_haircut(
    returns: pd.Series,
    haircut_bps_per_bar: float,
) -> pd.Series:
    """Subtract a constant per-bar cost (basis points) from simple returns."""
    s = _clean_returns(returns)
    return s - float(haircut_bps_per_bar) / 1.0e4


def path_wealth(paths: pd.DataFrame) -> pd.DataFrame:
    """Cumulative wealth starting at 1.0 (simple returns, columns = paths)."""
    return (1.0 + paths.astype(float)).cumprod()


def terminal_simple_return(paths: pd.DataFrame) -> pd.Series:
    """``W_H - 1`` for each path."""
    return path_wealth(paths).iloc[-1] - 1.0


def horizon_ev(
    paths: pd.DataFrame,
    *,
    level: float = 0.95,
) -> dict[str, float]:
    """Mean terminal simple return and percentile CI across simulated paths."""
    if paths is None or paths.empty:
        return {
            "mean_terminal": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_paths": 0,
        }
    term = terminal_simple_return(paths)
    alpha = (1.0 - float(level)) / 2.0
    return {
        "mean_terminal": float(term.mean()),
        "ci_low": float(term.quantile(alpha)),
        "ci_high": float(term.quantile(1.0 - alpha)),
        "n_paths": int(term.shape[0]),
    }


def p_not_beat_spy(
    strategy_paths: pd.DataFrame,
    spy_paths: pd.DataFrame,
) -> float:
    """``P(W_strategy <= W_spy)`` on paired columns (joint paths required)."""
    w_s = path_wealth(strategy_paths).iloc[-1]
    w_b = path_wealth(spy_paths).iloc[-1]
    a, b = w_s.align(w_b, join="inner")
    if a.empty:
        return float("nan")
    return float((a.to_numpy() <= b.to_numpy()).mean())


def p_not_beat_spy_from_joint(simulations: pd.DataFrame) -> float:
    """``P(not beat SPY)`` from a joint ``(simulation, asset)`` frame."""
    strat, spy = split_joint_simulations(simulations)
    return p_not_beat_spy(strat, spy)


def cvar(values: pd.Series | np.ndarray, *, alpha: float = 0.05) -> float:
    """Lower CVaR: mean of observations at or below the ``alpha`` quantile."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    q = float(np.quantile(x, alpha))
    tail = x[x <= q]
    if tail.size == 0:
        return float(q)
    return float(tail.mean())


def underwater_probs(paths: pd.DataFrame) -> dict[str, float]:
    """Probability wealth dips below 1.0, and probability terminal wealth < 1."""
    if paths is None or paths.empty:
        return {"p_ever_underwater": float("nan"), "p_terminal_underwater": float("nan")}
    w = path_wealth(paths)
    return {
        "p_ever_underwater": float((w.min(axis=0) < 1.0).mean()),
        "p_terminal_underwater": float((w.iloc[-1] < 1.0).mean()),
    }


def simulate_joint_paths(
    frame: pd.DataFrame,
    *,
    n_simulations: int,
    horizon: int,
    mean_block_length: float = 10.0,
    random_seed: int | None = 0,
    leverage: float = 1.0,
) -> pd.DataFrame:
    """Fit a joint stationary bootstrap and draw ``horizon``-bar paths."""
    scaled = scale_simple_returns(frame, leverage)
    sim = StationaryBlockBootstrap(
        n_simulations,
        horizon=horizon,
        random_seed=random_seed,
        mean_block_length=mean_block_length,
    )
    sim.fit(scaled)
    return sim.simulate(horizon)


# Re-exports used by notebooks / tests
__all__ = [
    "apply_cost_haircut",
    "asset_paths",
    "bootstrap_mean_distribution",
    "bootstrap_mean_inference",
    "cvar",
    "ev_significance",
    "excess_returns",
    "hac_mean_inference",
    "horizon_ev",
    "is_joint_simulations",
    "newey_west_lag",
    "newey_west_mean_se",
    "p_not_beat_spy",
    "p_not_beat_spy_from_joint",
    "path_wealth",
    "scale_simple_returns",
    "simulate_joint_paths",
    "split_joint_simulations",
    "terminal_simple_return",
    "underwater_probs",
]
