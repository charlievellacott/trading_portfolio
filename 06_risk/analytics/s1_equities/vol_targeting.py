"""Point-in-time portfolio volatility targeting (live-safe; no backtest imports).

Monday pre-open contract
------------------------
Pass only completed period returns whose exit is strictly before the decision
open. For S1 ``mon_open_mon_open``, that means history through entry ``t-2``
(the week entered at ``t-1`` exits at the open of ``t`` and is not yet known).
For ``mon_open_fri_close``, the prior week exits Friday and is usable.

Live and backtest must call the same scalar APIs
(``leverage_from_history`` / ``update_bayes_vol_state``) so train/serve agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import invgamma

ESTIMATOR_ROLLING = "rolling"
ESTIMATOR_EWM = "ewm"
ESTIMATOR_BAYES = "bayes"
VALID_ESTIMATORS = frozenset(
    {ESTIMATOR_ROLLING, ESTIMATOR_EWM, ESTIMATOR_BAYES}
)

DEFAULT_TARGET_ANN_VOL = 0.10
DEFAULT_FORGET = 0.94
DEFAULT_MIN_PERIODS = 13
DEFAULT_MIN_LEVERAGE = 0.25
DEFAULT_MAX_LEVERAGE = 1.50
DEFAULT_PERIODS_PER_YEAR = 52.0
DEFAULT_VOL_FLOOR = 1e-6
DEFAULT_BAYES_ALPHA0 = 2.0
DEFAULT_BAYES_BETA0 = 0.0


@dataclass(frozen=True)
class VolTargetConfig:
    """Volatility-targeting overlay configuration."""

    enabled: bool = True
    target_ann_vol: float = DEFAULT_TARGET_ANN_VOL
    estimator: str = ESTIMATOR_BAYES
    window: int | None = 26
    halflife: float | None = 10.0
    forget: float = DEFAULT_FORGET
    quantile: float = 0.5
    min_periods: int = DEFAULT_MIN_PERIODS
    warmup_leverage: float = 1.0
    min_leverage: float = DEFAULT_MIN_LEVERAGE
    max_leverage: float = DEFAULT_MAX_LEVERAGE
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR
    vol_floor: float = DEFAULT_VOL_FLOOR
    deadband: float = 0.0
    alpha0: float = DEFAULT_BAYES_ALPHA0
    beta0: float = DEFAULT_BAYES_BETA0

    def label(self) -> str:
        if not self.enabled:
            return "none"
        return vol_target_star(self)


@dataclass(frozen=True)
class BayesVolState:
    """Incremental inverse-gamma state (JSON-serializable floats)."""

    alpha: float
    beta: float
    n_obs: int = 0


def _validate_cfg(cfg: VolTargetConfig) -> None:
    if cfg.estimator not in VALID_ESTIMATORS:
        raise ValueError(f"unknown estimator={cfg.estimator!r}")
    if cfg.target_ann_vol <= 0 or not np.isfinite(cfg.target_ann_vol):
        raise ValueError("target_ann_vol must be positive")
    if cfg.min_leverage < 0 or cfg.max_leverage < cfg.min_leverage:
        raise ValueError("invalid leverage bounds")
    if cfg.min_periods < 1:
        raise ValueError("min_periods must be >= 1")
    if cfg.deadband < 0:
        raise ValueError("deadband must be >= 0")
    if not (0.0 < cfg.quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    if cfg.estimator == ESTIMATOR_ROLLING and (
        cfg.window is None or int(cfg.window) < 1
    ):
        raise ValueError("rolling estimator requires window >= 1")
    if cfg.estimator == ESTIMATOR_EWM and (
        cfg.halflife is None or float(cfg.halflife) <= 0
    ):
        raise ValueError("ewm estimator requires halflife > 0")
    if cfg.estimator == ESTIMATOR_BAYES and not (0.0 < cfg.forget <= 1.0):
        raise ValueError("bayes forget must be in (0, 1]")


def vol_target_star(cfg: VolTargetConfig) -> str:
    """Serialize config to a star string for notebooks / live config."""
    if not cfg.enabled:
        return "none"
    parts: list[str] = ["vt", cfg.estimator]
    if cfg.estimator == ESTIMATOR_ROLLING:
        parts.append(str(int(cfg.window)))
    elif cfg.estimator == ESTIMATOR_EWM:
        parts.append(f"{float(cfg.halflife):g}")
    else:
        parts.append(f"{float(cfg.forget):g}")
    # Target as percent without trailing zeros noise (10 for 0.10, 12 for 0.12)
    tgt_pct = cfg.target_ann_vol * 100.0
    parts.append(f"{tgt_pct:g}")
    parts.append(f"q{cfg.quantile:g}")
    if cfg.deadband and cfg.deadband > 0:
        parts.append(f"db{cfg.deadband:g}")
    return "_".join(parts)


def parse_vol_target_star(star: str) -> VolTargetConfig:
    """
    Parse star strings.

    Examples
    --------
    ``none``
    ``vt_bayes_0.94_10_q0.5``
    ``vt_bayes_0.94_10_q0.5_db0.05``
    ``vt_rolling_26_10_q0.5``
    ``vt_ewm_10_12_q0.75``
    """
    s = str(star).strip().lower()
    if s in {"none", "off", "disabled"}:
        return VolTargetConfig(enabled=False)
    if not s.startswith("vt_"):
        raise ValueError(f"unrecognized VT_STAR={star!r}")
    body = s[3:]
    deadband = 0.0
    if "_db" in body:
        body, db_str = body.rsplit("_db", 1)
        deadband = float(db_str)
    if "_q" not in body:
        raise ValueError(f"expected ..._q<quantile> in {star!r}")
    body, q_str = body.rsplit("_q", 1)
    quantile = float(q_str)
    parts = body.split("_")
    if len(parts) < 3:
        raise ValueError(f"expected vt_<estimator>_<param>_<target>, got {star!r}")
    estimator = parts[0]
    if estimator not in VALID_ESTIMATORS:
        raise ValueError(f"unknown estimator in {star!r}")
    param = float(parts[1])
    target_ann_vol = float(parts[2]) / 100.0
    window = int(param) if estimator == ESTIMATOR_ROLLING else 26
    halflife = float(param) if estimator == ESTIMATOR_EWM else 10.0
    forget = float(param) if estimator == ESTIMATOR_BAYES else DEFAULT_FORGET
    cfg = VolTargetConfig(
        enabled=True,
        estimator=estimator,
        window=window,
        halflife=halflife,
        forget=forget,
        target_ann_vol=target_ann_vol,
        quantile=quantile,
        deadband=deadband,
    )
    _validate_cfg(cfg)
    return cfg


def apply_deadband(prev: float | None, new: float, cfg: VolTargetConfig) -> float:
    """Keep ``prev`` when relative change is within ``cfg.deadband``."""
    if prev is None or not np.isfinite(prev) or cfg.deadband <= 0:
        return float(new)
    if not np.isfinite(new):
        return float(prev)
    base = abs(float(prev))
    if base < 1e-12:
        return float(new)
    if abs(float(new) - float(prev)) / base <= float(cfg.deadband):
        return float(prev)
    return float(new)


def _clip_leverage(raw: float, cfg: VolTargetConfig) -> float:
    return float(np.clip(raw, cfg.min_leverage, cfg.max_leverage))


def _ann_vol_from_var(var: float, cfg: VolTargetConfig) -> float:
    if not np.isfinite(var) or var <= 0:
        return np.nan
    return float(np.sqrt(var) * np.sqrt(cfg.periods_per_year))


def _leverage_from_vol(vol_ann: float, cfg: VolTargetConfig) -> float:
    if not np.isfinite(vol_ann) or vol_ann < cfg.vol_floor:
        return float(cfg.warmup_leverage)
    return _clip_leverage(cfg.target_ann_vol / max(vol_ann, cfg.vol_floor), cfg)


def initial_bayes_vol_state(cfg: VolTargetConfig) -> BayesVolState:
    """Prior state before any observations."""
    return BayesVolState(alpha=float(cfg.alpha0), beta=float(cfg.beta0), n_obs=0)


def update_bayes_vol_state(
    state: BayesVolState,
    r: float,
    cfg: VolTargetConfig,
) -> BayesVolState:
    """Forgetful inverse-gamma update with one completed period return."""
    if not np.isfinite(r):
        return state
    d = float(cfg.forget)
    alpha = d * float(state.alpha) + 0.5
    beta = d * float(state.beta) + 0.5 * float(r) ** 2
    return BayesVolState(alpha=alpha, beta=beta, n_obs=int(state.n_obs) + 1)


def vol_from_bayes_state(state: BayesVolState, cfg: VolTargetConfig) -> float:
    """Annualized vol from IG posterior (mean if quantile≈0.5, else ppf)."""
    a = float(state.alpha)
    b = float(state.beta)
    if state.n_obs < cfg.min_periods or a <= 1.0 or b < 0 or not np.isfinite(a + b):
        return np.nan
    q = float(cfg.quantile)
    if abs(q - 0.5) < 1e-12 and a > 1.0:
        # Posterior mean of variance for InvGamma(a, scale=b)
        var = b / (a - 1.0)
    else:
        var = float(invgamma.ppf(q, a=a, scale=b))
    return _ann_vol_from_var(var, cfg)


def rolling_vol_series(
    returns: pd.Series,
    *,
    window: int,
    min_periods: int,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.Series:
    """
    Trailing sample stdev annualized; value at ``i`` uses returns ``[:i]`` only
    (excludes the contemporaneous return).
    """
    r = returns.astype(float)
    out = pd.Series(np.nan, index=r.index, dtype=float)
    vals = r.to_numpy(dtype=float)
    for i in range(len(vals)):
        hist = vals[:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_periods:
            continue
        windowed = hist[-window:] if len(hist) >= window else hist
        if len(windowed) < min_periods:
            continue
        sd = float(np.std(windowed, ddof=1))
        if np.isfinite(sd):
            out.iloc[i] = sd * np.sqrt(periods_per_year)
    return out


def ewm_vol_series(
    returns: pd.Series,
    *,
    halflife: float,
    min_periods: int,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.Series:
    """EWM variance recursion; value at ``i`` uses returns ``[:i]`` only."""
    r = returns.astype(float)
    out = pd.Series(np.nan, index=r.index, dtype=float)
    lam = 0.5 ** (1.0 / float(halflife))
    sigma2 = np.nan
    n = 0
    vals = r.to_numpy(dtype=float)
    for i, ri in enumerate(vals):
        # Emit estimate *before* incorporating return i
        if n >= min_periods and np.isfinite(sigma2) and sigma2 > 0:
            out.iloc[i] = float(np.sqrt(sigma2) * np.sqrt(periods_per_year))
        if np.isfinite(ri):
            if not np.isfinite(sigma2):
                sigma2 = float(ri) ** 2
            else:
                sigma2 = lam * sigma2 + (1.0 - lam) * float(ri) ** 2
            n += 1
    return out


def bayes_vol_series(
    returns: pd.Series,
    cfg: VolTargetConfig,
) -> pd.Series:
    """Bayesian IG vol; value at ``i`` uses returns ``[:i]`` only."""
    r = returns.astype(float)
    out = pd.Series(np.nan, index=r.index, dtype=float)
    state = initial_bayes_vol_state(cfg)
    for i, ri in enumerate(r.to_numpy(dtype=float)):
        out.iloc[i] = vol_from_bayes_state(state, cfg)
        state = update_bayes_vol_state(state, float(ri) if np.isfinite(ri) else np.nan, cfg)
    return out


def vol_estimate_series(returns: pd.Series, cfg: VolTargetConfig) -> pd.Series:
    """Dispatch trailing annualized vol estimator (PIT: excludes current bar)."""
    _validate_cfg(cfg)
    r = returns.astype(float).sort_index()
    if cfg.estimator == ESTIMATOR_ROLLING:
        return rolling_vol_series(
            r,
            window=int(cfg.window),
            min_periods=cfg.min_periods,
            periods_per_year=cfg.periods_per_year,
        )
    if cfg.estimator == ESTIMATOR_EWM:
        return ewm_vol_series(
            r,
            halflife=float(cfg.halflife),
            min_periods=cfg.min_periods,
            periods_per_year=cfg.periods_per_year,
        )
    return bayes_vol_series(r, cfg)


def leverage_series(returns: pd.Series, cfg: VolTargetConfig) -> pd.DataFrame:
    """
    Vectorized research path: vol estimate, raw leverage, deadband-smoothed leverage.

    Index aligned to ``returns``. Row ``i`` uses only returns before ``i``.
    """
    _validate_cfg(cfg)
    r = returns.astype(float).sort_index()
    if not cfg.enabled:
        ones = pd.Series(1.0, index=r.index, dtype=float)
        return pd.DataFrame(
            {"vol_est": np.nan, "raw_leverage": ones, "leverage": ones},
            index=r.index,
        )
    vol = vol_estimate_series(r, cfg)
    raw = vol.map(lambda v: _leverage_from_vol(float(v), cfg) if np.isfinite(v) else float(cfg.warmup_leverage))
    lev_vals: list[float] = []
    prev: float | None = None
    for v in raw.to_numpy(dtype=float):
        applied = apply_deadband(prev, float(v), cfg)
        lev_vals.append(applied)
        prev = applied
    return pd.DataFrame(
        {
            "vol_est": vol,
            "raw_leverage": raw.astype(float),
            "leverage": pd.Series(lev_vals, index=r.index, dtype=float),
        },
        index=r.index,
    )


def leverage_from_history(
    past_returns: pd.Series | list[float] | np.ndarray,
    cfg: VolTargetConfig,
    *,
    prev_leverage: float | None = None,
) -> float:
    """
    Live / backtest scalar entry: leverage from completed PIT-safe returns only.

    ``past_returns`` must already exclude any period whose exit is not yet known.
    """
    _validate_cfg(cfg)
    if not cfg.enabled:
        return 1.0
    if isinstance(past_returns, pd.Series):
        hist = past_returns.astype(float).dropna()
    else:
        hist = pd.Series(np.asarray(past_returns, dtype=float)).dropna()
    if len(hist) < cfg.min_periods:
        return apply_deadband(prev_leverage, float(cfg.warmup_leverage), cfg)

    if cfg.estimator == ESTIMATOR_BAYES:
        state = initial_bayes_vol_state(cfg)
        for ri in hist.to_numpy(dtype=float):
            state = update_bayes_vol_state(state, float(ri), cfg)
        vol = vol_from_bayes_state(state, cfg)
    elif cfg.estimator == ESTIMATOR_ROLLING:
        window = int(cfg.window)
        windowed = hist.iloc[-window:] if len(hist) >= window else hist
        sd = float(windowed.std(ddof=1))
        vol = sd * np.sqrt(cfg.periods_per_year) if np.isfinite(sd) else np.nan
    else:
        lam = 0.5 ** (1.0 / float(cfg.halflife))
        sigma2 = np.nan
        for ri in hist.to_numpy(dtype=float):
            if not np.isfinite(sigma2):
                sigma2 = float(ri) ** 2
            else:
                sigma2 = lam * sigma2 + (1.0 - lam) * float(ri) ** 2
        vol = (
            float(np.sqrt(sigma2) * np.sqrt(cfg.periods_per_year))
            if np.isfinite(sigma2) and sigma2 > 0
            else np.nan
        )

    raw = _leverage_from_vol(float(vol), cfg) if np.isfinite(vol) else float(cfg.warmup_leverage)
    return apply_deadband(prev_leverage, raw, cfg)


def compute_gross_leverage(
    past_returns: pd.Series | list[float] | np.ndarray,
    vol_cfg: VolTargetConfig,
    *,
    ic_multiplier: float = 1.0,
    prev_leverage: float | None = None,
    min_leverage: float | None = None,
    max_gross: float | None = None,
) -> float:
    """
    Live one-call helper: ``clip(L_vol * m_ic, min, max)``.

    Caller supplies a PIT-safe return history and an optional IC multiplier
    (from ``risk.analytics.s1_equities.signal_conviction.ic_multiplier_from_history``, or 1.0).
    """
    l_vol = leverage_from_history(
        past_returns, vol_cfg, prev_leverage=prev_leverage
    )
    lo = float(vol_cfg.min_leverage if min_leverage is None else min_leverage)
    hi = float(vol_cfg.max_leverage if max_gross is None else max_gross)
    m = float(ic_multiplier) if np.isfinite(ic_multiplier) else 1.0
    return float(np.clip(l_vol * m, lo, hi))
