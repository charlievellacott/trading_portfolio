"""Bayesian-shrunk IC conviction sizing (live-safe; no backtest imports).

Caller builds the weekly IC series (e.g. via ``date_ic_series``) with the same
PIT lag as volatility targeting: only weeks whose label is known before the
Monday open may enter ``past_ic``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FORGET = 0.94
DEFAULT_PRIOR_IC = 0.0
DEFAULT_PRIOR_VAR = 0.01
DEFAULT_MIN_MULT = 0.25
DEFAULT_MAX_MULT = 1.25
DEFAULT_MIN_PERIODS = 13
DEFAULT_WARMUP = 1.0
FISHER_EPS = 1e-6


@dataclass(frozen=True)
class ICScaleConfig:
    """IC conviction multiplier configuration."""

    enabled: bool = True
    k: float = 25.0
    forget: float = DEFAULT_FORGET
    prior_ic: float = DEFAULT_PRIOR_IC
    prior_var: float = DEFAULT_PRIOR_VAR
    min_mult: float = DEFAULT_MIN_MULT
    max_mult: float = DEFAULT_MAX_MULT
    min_periods: int = DEFAULT_MIN_PERIODS
    warmup_multiplier: float = DEFAULT_WARMUP

    def label(self) -> str:
        if not self.enabled:
            return "none"
        return ic_scale_star(self)


@dataclass(frozen=True)
class BayesICState:
    """Gaussian conjugate state on Fisher-z IC (JSON-serializable)."""

    mean_z: float
    var_z: float
    n_obs: int = 0


def _validate_cfg(cfg: ICScaleConfig) -> None:
    if cfg.k <= 0 or not np.isfinite(cfg.k):
        raise ValueError("k must be positive")
    if not (0.0 < cfg.forget <= 1.0):
        raise ValueError("forget must be in (0, 1]")
    if cfg.prior_var <= 0:
        raise ValueError("prior_var must be positive")
    if cfg.min_mult < 0 or cfg.max_mult < cfg.min_mult:
        raise ValueError("invalid multiplier bounds")
    if cfg.min_periods < 1:
        raise ValueError("min_periods must be >= 1")


def ic_scale_star(cfg: ICScaleConfig) -> str:
    """Serialize config: ``none`` or ``ic_k25_0.94_f0.25_c1.25``."""
    if not cfg.enabled:
        return "none"
    return (
        f"ic_k{cfg.k:g}_{cfg.forget:g}_f{cfg.min_mult:g}_c{cfg.max_mult:g}"
    )


def parse_ic_scale_star(star: str) -> ICScaleConfig:
    """Parse ``none`` / ``ic_k25_0.94_f0.25_c1.25``."""
    s = str(star).strip().lower()
    if s in {"none", "off", "disabled"}:
        return ICScaleConfig(enabled=False)
    if not s.startswith("ic_"):
        raise ValueError(f"unrecognized IC_STAR={star!r}")
    body = s[3:]
    parts = body.split("_")
    if len(parts) != 4 or not parts[0].startswith("k"):
        raise ValueError(f"expected ic_k<k>_<forget>_f<floor>_c<cap>, got {star!r}")
    k = float(parts[0][1:])
    forget = float(parts[1])
    if not parts[2].startswith("f") or not parts[3].startswith("c"):
        raise ValueError(f"expected floor/cap markers in {star!r}")
    min_mult = float(parts[2][1:])
    max_mult = float(parts[3][1:])
    cfg = ICScaleConfig(
        enabled=True,
        k=k,
        forget=forget,
        min_mult=min_mult,
        max_mult=max_mult,
    )
    _validate_cfg(cfg)
    return cfg


def fisher_z(ic: float) -> float:
    """Artanh of clipped IC."""
    x = float(np.clip(ic, -1.0 + FISHER_EPS, 1.0 - FISHER_EPS))
    return float(np.arctanh(x))


def inv_fisher_z(z: float) -> float:
    return float(np.tanh(z))


def obs_var_fisher(n_names: float) -> float:
    """Approx sampling variance of Fisher-z Spearman IC."""
    n = float(n_names)
    if not np.isfinite(n) or n <= 4:
        return np.nan
    return 1.0 / (n - 3.0)


def initial_bayes_ic_state(cfg: ICScaleConfig) -> BayesICState:
    return BayesICState(
        mean_z=fisher_z(cfg.prior_ic),
        var_z=float(cfg.prior_var),
        n_obs=0,
    )


def update_bayes_ic_state(
    state: BayesICState,
    ic: float,
    n_names: float,
    cfg: ICScaleConfig,
) -> BayesICState:
    """Forgetful Gaussian conjugate update on Fisher-z."""
    if not np.isfinite(ic):
        return state
    r_var = obs_var_fisher(n_names)
    if not np.isfinite(r_var) or r_var <= 0:
        return state
    d = float(cfg.forget)
    # Discount prior precision / inflate variance
    prior_var = float(state.var_z) / d if d < 1.0 else float(state.var_z)
    prior_mean = float(state.mean_z)
    z = fisher_z(ic)
    post_var = 1.0 / (1.0 / prior_var + 1.0 / r_var)
    post_mean = post_var * (prior_mean / prior_var + z / r_var)
    return BayesICState(
        mean_z=float(post_mean),
        var_z=float(post_var),
        n_obs=int(state.n_obs) + 1,
    )


def ic_from_state(state: BayesICState) -> float:
    return inv_fisher_z(float(state.mean_z))


def multiplier_from_ic(ic_post: float, cfg: ICScaleConfig) -> float:
    """``clip(k * ic_post, floor, cap)``; negative IC clips to floor (no invert)."""
    if not cfg.enabled:
        return 1.0
    if not np.isfinite(ic_post):
        return float(cfg.warmup_multiplier)
    raw = float(cfg.k) * float(ic_post)
    return float(np.clip(raw, cfg.min_mult, cfg.max_mult))


def bayes_ic_series(
    ic: pd.Series,
    n_names: pd.Series,
    cfg: ICScaleConfig,
) -> pd.DataFrame:
    """
    Posterior IC and multiplier; row ``i`` uses observations strictly before ``i``.
    """
    _validate_cfg(cfg)
    ics = ic.astype(float).sort_index()
    ns = n_names.reindex(ics.index).astype(float)
    state = initial_bayes_ic_state(cfg)
    post: list[float] = []
    mult: list[float] = []
    for dt in ics.index:
        if state.n_obs < cfg.min_periods:
            post.append(np.nan)
            mult.append(float(cfg.warmup_multiplier) if cfg.enabled else 1.0)
        else:
            ic_hat = ic_from_state(state)
            post.append(ic_hat)
            mult.append(multiplier_from_ic(ic_hat, cfg) if cfg.enabled else 1.0)
        state = update_bayes_ic_state(
            state,
            float(ics.loc[dt]),
            float(ns.loc[dt]) if np.isfinite(ns.loc[dt]) else np.nan,
            cfg,
        )
    return pd.DataFrame(
        {"ic_posterior": post, "ic_multiplier": mult},
        index=ics.index,
    )


def ic_multiplier_from_history(
    past_ic: pd.Series | list[float] | np.ndarray,
    past_n_names: pd.Series | list[float] | np.ndarray,
    cfg: ICScaleConfig,
) -> float:
    """Live/backtest scalar: multiplier from completed PIT-safe IC history."""
    _validate_cfg(cfg)
    if not cfg.enabled:
        return 1.0
    if isinstance(past_ic, pd.Series):
        ics = past_ic.astype(float)
    else:
        ics = pd.Series(np.asarray(past_ic, dtype=float))
    if isinstance(past_n_names, pd.Series):
        ns = past_n_names.astype(float).reindex(ics.index)
        if ns.isna().all() and len(ns) == len(ics):
            ns = pd.Series(np.asarray(past_n_names, dtype=float), index=ics.index)
    else:
        ns = pd.Series(np.asarray(past_n_names, dtype=float), index=ics.index)
    # Align lengths
    n = min(len(ics), len(ns))
    ics = ics.iloc[:n]
    ns = ns.iloc[:n]
    state = initial_bayes_ic_state(cfg)
    for i in range(n):
        state = update_bayes_ic_state(
            state,
            float(ics.iloc[i]),
            float(ns.iloc[i]),
            cfg,
        )
    if state.n_obs < cfg.min_periods:
        return float(cfg.warmup_multiplier)
    return multiplier_from_ic(ic_from_state(state), cfg)


def ic_posterior_from_history(
    past_ic: pd.Series | list[float] | np.ndarray,
    past_n_names: pd.Series | list[float] | np.ndarray,
    cfg: ICScaleConfig,
) -> float:
    """Posterior mean IC from completed history (diagnostic / live logging)."""
    _validate_cfg(cfg)
    if isinstance(past_ic, pd.Series):
        ics = past_ic.astype(float)
    else:
        ics = pd.Series(np.asarray(past_ic, dtype=float))
    if isinstance(past_n_names, pd.Series):
        ns = past_n_names.astype(float)
        if len(ns) != len(ics):
            ns = pd.Series(np.asarray(past_n_names, dtype=float), index=ics.index)
    else:
        ns = pd.Series(np.asarray(past_n_names, dtype=float), index=ics.index)
    n = min(len(ics), len(ns))
    state = initial_bayes_ic_state(cfg)
    for i in range(n):
        state = update_bayes_ic_state(
            state, float(ics.iloc[i]), float(ns.iloc[i]), cfg
        )
    if state.n_obs < cfg.min_periods:
        return np.nan
    return ic_from_state(state)
