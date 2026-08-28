"""PSR / DSR inference (Bailey & Lopez de Prado). No scipy dependency."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_EULER_GAMMA = 0.5772156649015329
_MIN_OBS = 4


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF via Newton-Raphson on ``_norm_cdf``."""
    q = float(p)
    if q <= 0.0:
        return float("-inf")
    if q >= 1.0:
        return float("inf")
    if q < 0.5:
        return -_norm_ppf(1.0 - q)
    x = math.sqrt(-2.0 * math.log(1.0 - q))
    for _ in range(12):
        err = _norm_cdf(x) - q
        if abs(err) < 1e-12:
            break
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        if pdf <= 0.0:
            break
        x -= err / pdf
    return x


def return_moments(
    ret: pd.Series,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Annualized Sharpe, n_obs, skew, Pearson kurtosis from a return series."""
    empty = {
        "sr": float("nan"),
        "n_obs": 0,
        "skew": float("nan"),
        "kurtosis": float("nan"),
    }
    if ret is None or ret.empty:
        return empty
    r = pd.to_numeric(ret, errors="coerce").astype(float).dropna()
    n = int(r.shape[0])
    if n < _MIN_OBS:
        return {**empty, "n_obs": n}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return {**empty, "n_obs": n}
    sr = float(math.sqrt(periods_per_year) * mu / sd)
    sk = float(r.skew())
    # pandas skew/kurt are Fisher-style for kurt; convert to Pearson gamma_4.
    ku = float(r.kurtosis() + 3.0)
    return {"sr": sr, "n_obs": n, "skew": sk, "kurtosis": ku}


def sharpe_std_error(
    sr: float,
    n: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Asymptotic std dev of Sharpe estimate (Bailey & Lopez de Prado)."""
    if n < _MIN_OBS or not np.isfinite(sr):
        return float("nan")
    sk = 0.0 if not np.isfinite(skew) else float(skew)
    ku = 3.0 if not np.isfinite(kurtosis) else float(kurtosis)
    inner = 1.0 - sk * float(sr) + ((ku - 1.0) / 4.0) * float(sr) ** 2
    if inner <= 0:
        return float("nan")
    return float(math.sqrt(inner / (n - 1)))


def probabilistic_sharpe_ratio(
    sr: float,
    n: int,
    skew: float,
    kurtosis: float,
    *,
    sr_benchmark: float = 1.0,
) -> float:
    """PSR = P(true SR > sr_benchmark)."""
    se = sharpe_std_error(sr, n, skew, kurtosis)
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    return _norm_cdf((float(sr) - float(sr_benchmark)) / se)


def expected_max_sharpe(
    n_trials: int,
    n: int,
    skew: float,
    kurtosis: float,
    *,
    sr_benchmark: float = 0.0,
) -> float:
    """Expected maximum Sharpe under null from n_trials independent searches."""
    if n_trials < 1 or n < _MIN_OBS:
        return float("nan")
    se = sharpe_std_error(0.0, n, skew, kurtosis)
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    if n_trials == 1:
        return float(sr_benchmark)
    z1 = _norm_ppf(1.0 - 1.0 / float(n_trials))
    z2 = _norm_ppf(1.0 - 1.0 / (float(n_trials) * math.e))
    sr0 = se * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)
    return float(sr0 + sr_benchmark)


def deflated_sharpe_ratio(
    sr: float,
    n: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    *,
    sr_benchmark: float = 0.0,
) -> float:
    """DSR = P(true SR > expected max SR from n_trials under null)."""
    if n_trials < 1:
        return float("nan")
    sr0 = expected_max_sharpe(
        n_trials, n, skew, kurtosis, sr_benchmark=sr_benchmark
    )
    se = sharpe_std_error(sr, n, skew, kurtosis)
    if not np.isfinite(se) or se <= 0 or not np.isfinite(sr0):
        return float("nan")
    return _norm_cdf((float(sr) - float(sr0)) / se)
