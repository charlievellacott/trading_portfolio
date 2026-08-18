"""S2 cointegration: Engle-Granger discovery, PIT hedges, z-score, half-life, health metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from data.processing.feature_implementation.kalman import kalman_linear_regression
from data.processing.feature_implementation.linear_regression import (
    rolling_ols_stats,
    rolling_residual,
)

COINT_PVALUE = 0.05


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------


def to_log_price(px: pd.Series) -> pd.Series:
    """``ln(price)``; non-positive / non-finite prices → NaN."""
    p = px.astype(float)
    out = pd.Series(np.nan, index=px.index, dtype=float)
    ok = p.notna() & np.isfinite(p) & (p > 0)
    out.loc[ok] = np.log(p.loc[ok].to_numpy(dtype=float))
    return out


# ---------------------------------------------------------------------------
# Discovery (non-PIT — train window only)
# ---------------------------------------------------------------------------


def is_integrated_order_one(
    s: pd.Series,
    *,
    pvalue_threshold: float = COINT_PVALUE,
    autolag: str = "aic",
) -> tuple[bool, float, float]:
    """I(1) screen: unit root in levels (fail to reject) and stationary differences (reject)."""
    arr = s.astype(float).to_numpy()
    finite = arr[np.isfinite(arr)]
    if len(finite) < 20:
        return False, np.nan, np.nan

    try:
        level_p = float(adfuller(finite, autolag=autolag)[1])
    except (ValueError, np.linalg.LinAlgError):
        return False, np.nan, np.nan

    diffs = np.diff(finite)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 20:
        return False, level_p, np.nan

    try:
        diff_p = float(adfuller(diffs, autolag=autolag)[1])
    except (ValueError, np.linalg.LinAlgError):
        return False, level_p, np.nan

    is_i1 = (level_p >= pvalue_threshold) and (diff_p < pvalue_threshold)
    return is_i1, level_p, diff_p


@dataclass(frozen=True)
class CointResult:
    """Engle-Granger discovery result (non-PIT; not for use inside a backtest loop)."""

    is_cointegrated: bool
    pvalue: float
    tstat: float
    crit_values: dict[str, float]
    direction: str
    alpha: float
    beta: float
    y_is_i1: bool
    x_is_i1: bool
    y_adf_level_p: float
    y_adf_diff_p: float
    x_adf_level_p: float
    x_adf_diff_p: float


def test_cointegration(
    y: pd.Series,
    x: pd.Series,
    *,
    pvalue_threshold: float = COINT_PVALUE,
    trend: str = "c",
    autolag: str = "aic",
) -> CointResult:
    """Discovery-only Engle-Granger on a train window; never call inside a backtest loop.

    Tests both ``y~x`` and ``x~y``, keeps the lower p-value, then OLS for alpha/beta
    (``coint`` itself does not return the hedge). ``pvalue`` is a float for H-010 sizing.
    """
    if len(y) != len(x):
        raise ValueError("y and x must have the same length")
    if not y.index.equals(x.index):
        raise ValueError("y and x must share an identical index")

    y_i1, y_lvl, y_diff = is_integrated_order_one(
        y, pvalue_threshold=pvalue_threshold, autolag=autolag
    )
    x_i1, x_lvl, x_diff = is_integrated_order_one(
        x, pvalue_threshold=pvalue_threshold, autolag=autolag
    )
    both_i1 = y_i1 and x_i1

    yv = y.astype(float)
    xv = x.astype(float)
    mask = yv.notna() & xv.notna() & np.isfinite(yv) & np.isfinite(xv)
    y_c = yv.loc[mask]
    x_c = xv.loc[mask]

    nan_result = CointResult(
        is_cointegrated=False,
        pvalue=np.nan,
        tstat=np.nan,
        crit_values={},
        direction="y~x",
        alpha=np.nan,
        beta=np.nan,
        y_is_i1=y_i1,
        x_is_i1=x_i1,
        y_adf_level_p=y_lvl,
        y_adf_diff_p=y_diff,
        x_adf_level_p=x_lvl,
        x_adf_diff_p=x_diff,
    )

    if len(y_c) < 20:
        return nan_result

    try:
        t_yx, p_yx, crit_yx = coint(y_c, x_c, trend=trend, autolag=autolag)
        t_xy, p_xy, crit_xy = coint(x_c, y_c, trend=trend, autolag=autolag)
    except (ValueError, np.linalg.LinAlgError):
        return nan_result

    if float(p_yx) <= float(p_xy):
        direction = "y~x"
        tstat = float(t_yx)
        pvalue = float(p_yx)
        crit_raw = crit_yx
        dep, indep = y_c, x_c
    else:
        direction = "x~y"
        tstat = float(t_xy)
        pvalue = float(p_xy)
        crit_raw = crit_xy
        dep, indep = x_c, y_c

    # statsmodels crit_values is array [1%, 5%, 10%]
    if isinstance(crit_raw, dict):
        crit_values = {str(k): float(v) for k, v in crit_raw.items()}
    else:
        crit_arr = np.asarray(crit_raw, dtype=float).ravel()
        labels = ("1%", "5%", "10%")
        crit_values = {
            labels[i]: float(crit_arr[i]) for i in range(min(len(labels), len(crit_arr)))
        }

    alpha, beta = np.nan, np.nan
    try:
        design = add_constant(indep.to_numpy(dtype=float), has_constant="add")
        fit = OLS(dep.to_numpy(dtype=float), design).fit()
        alpha = float(fit.params[0])
        beta = float(fit.params[1])
    except (ValueError, np.linalg.LinAlgError):
        pass

    is_coint = bool(both_i1 and np.isfinite(pvalue) and pvalue < pvalue_threshold)
    return CointResult(
        is_cointegrated=is_coint,
        pvalue=pvalue,
        tstat=tstat,
        crit_values=crit_values,
        direction=direction,
        alpha=alpha,
        beta=beta,
        y_is_i1=y_i1,
        x_is_i1=x_i1,
        y_adf_level_p=y_lvl,
        y_adf_diff_p=y_diff,
        x_adf_level_p=x_lvl,
        x_adf_diff_p=x_diff,
    )


# pytest would otherwise collect this as a test (name starts with test_)
test_cointegration.__test__ = False


# ---------------------------------------------------------------------------
# PIT spread layer
# ---------------------------------------------------------------------------


def rolling_hedge(y: pd.Series, x: pd.Series, *, window: int = 252) -> pd.DataFrame:
    """Rolling OLS hedge; spread ``s_t = y_t - alpha_t - beta_t * x_t`` (window ends at t inclusive)."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")
    if len(y) != len(x):
        raise ValueError("y and x must have the same length")
    if not y.index.equals(x.index):
        raise ValueError("y and x must share an identical index")

    stats = rolling_ols_stats(y.astype(float), x.astype(float), window)
    spread = rolling_residual(y, x, stats["alpha"], stats["beta"])
    return pd.DataFrame(
        {"alpha": stats["alpha"], "beta": stats["beta"], "spread": spread},
        index=y.index,
    )


def kalman_hedge(
    y: pd.Series,
    x: pd.Series,
    *,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
    burn_in: int = 30,
) -> pd.DataFrame:
    """Kalman hedge wrapper; see ``kalman_linear_regression`` for spread / spread_var / z_innov."""
    return kalman_linear_regression(
        y, x, delta=delta, obs_var=obs_var, burn_in=burn_in
    )


def rolling_zscore(
    spread: pd.Series,
    *,
    window: int = 60,
    ddof: int = 1,
) -> pd.Series:
    """Rolling ``(s - mean) / std`` of the spread; NaN when std==0; no bfill."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")
    s = spread.astype(float)
    mu = s.rolling(window, min_periods=window).mean()
    sigma = s.rolling(window, min_periods=window).std(ddof=ddof)
    z = (s - mu) / sigma
    return z.where(sigma > 0)


def ewm_zscore(
    spread: pd.Series,
    *,
    span: int = 60,
) -> pd.Series:
    """EWM ``(s - mean) / std`` of the spread; NaN when std==0; no bfill."""
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span!r}")
    s = spread.astype(float)
    mu = s.ewm(span=span, min_periods=span, adjust=False).mean()
    sigma = s.ewm(span=span, min_periods=span, adjust=False).std()
    z = (s - mu) / sigma
    return z.where(sigma > 0)


def ou_residual_score(spread: pd.Series, *, window: int = 60) -> pd.Series:
    """Rolling AR(1) leftover of ``s`` in residual-sigma units (PIT window ending at t)."""
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window!r}")
    s = spread.astype(float).to_numpy()
    n = len(s)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = s[i - window + 1 : i + 1]
        if not np.all(np.isfinite(w)):
            continue
        lag = w[:-1]
        nxt = w[1:]
        design = np.column_stack([np.ones(len(lag)), lag])
        try:
            coef, _, rank, _ = np.linalg.lstsq(design, nxt, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < 2:
            continue
        c = float(coef[0])
        phi = float(coef[1])
        if not np.isfinite(phi) or abs(phi) >= 1.0 - 1e-12:
            continue
        mu = c / (1.0 - phi)
        resid = nxt - (c + phi * lag)
        sig = float(np.std(resid, ddof=1)) if len(resid) > 1 else float("nan")
        if (not np.isfinite(sig)) or sig <= 0.0:
            continue
        out[i] = (w[-1] - mu) / sig
    return pd.Series(out, index=spread.index, dtype=float)


def adaptive_zscore(
    spread: pd.Series,
    half_life: pd.Series,
    *,
    z_min: int = 20,
    z_max: int = 120,
    ddof: int = 1,
) -> pd.Series:
    """Trad z with ``z_window_t = clip(2 * half_life_{t-1}, z_min, z_max)``. No bfill."""
    if z_min < 2 or z_max < z_min:
        raise ValueError(f"need 2 <= z_min <= z_max, got {z_min!r}, {z_max!r}")
    s = spread.astype(float).to_numpy()
    hl = half_life.astype(float).to_numpy()
    n = len(s)
    out = np.full(n, np.nan)
    for i in range(n):
        if i == 0 or not np.isfinite(hl[i - 1]) or hl[i - 1] <= 0.0:
            continue
        w = int(round(2.0 * float(hl[i - 1])))
        w = max(int(z_min), min(int(z_max), w))
        if i + 1 < w:
            continue
        chunk = s[i - w + 1 : i + 1]
        if not np.all(np.isfinite(chunk)):
            continue
        mu = float(np.mean(chunk))
        sig = float(np.std(chunk, ddof=ddof))
        if (not np.isfinite(sig)) or sig <= 0.0:
            continue
        out[i] = (s[i] - mu) / sig
    return pd.Series(out, index=spread.index, dtype=float)


def _half_life_from_ar1(spread: np.ndarray) -> float:
    """Discrete OU half-life ``-ln(2)/ln(1+b)`` from ``Δs = a + b*s_{t-1}``; NaN if not mean-reverting."""
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return float("nan")
    lag = s[:-1]
    delta = np.diff(s)
    ok = np.isfinite(lag) & np.isfinite(delta)
    if int(ok.sum()) < 2:
        return float("nan")
    lag_v = lag[ok]
    d_v = delta[ok]
    design = np.column_stack([np.ones(len(lag_v)), lag_v])
    try:
        coef, _, rank, _ = np.linalg.lstsq(design, d_v, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    if rank < 2:
        return float("nan")
    b = float(coef[1])
    # Need b in (-1, 0): no MR / unit root near 0; oscillatory if 1+b <= 0
    if (not np.isfinite(b)) or b >= -1e-12 or (1.0 + b) <= 1e-12:
        return float("nan")
    hl = float(-np.log(2.0) / np.log(1.0 + b))
    if (not np.isfinite(hl)) or hl <= 0.0:
        return float("nan")
    return hl


def ou_half_life(spread: pd.Series) -> float:
    """Full-sample discrete OU half-life of the spread; NaN if no mean reversion."""
    return _half_life_from_ar1(spread.astype(float).to_numpy())


def rolling_ou_half_life(spread: pd.Series, *, window: int = 252) -> pd.Series:
    """Rolling discrete OU half-life of the spread; NaN until ``window`` bars or if not mean-reverting."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")
    s = spread.astype(float).to_numpy()
    n = len(s)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = _half_life_from_ar1(s[i - window + 1 : i + 1])
    return pd.Series(out, index=spread.index, dtype=float)


# ---------------------------------------------------------------------------
# Health metrics
# ---------------------------------------------------------------------------


def rolling_adf_pvalue(
    spread: pd.Series,
    *,
    window: int = 252,
    regression: str = "c",
    autolag: str = "aic",
) -> pd.Series:
    """Rolling ADF p-value of the spread (lower ⇒ more evidence of stationarity)."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")
    s = spread.astype(float).to_numpy()
    n = len(s)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = s[i - window + 1 : i + 1]
        finite = w[np.isfinite(w)]
        if len(finite) < window:
            continue
        try:
            out[i] = float(adfuller(finite, regression=regression, autolag=autolag)[1])
        except (ValueError, np.linalg.LinAlgError):
            continue
    return pd.Series(out, index=spread.index, dtype=float)


def residual_variance_ratio(
    spread: pd.Series,
    *,
    window: int = 60,
    baseline_window: int = 252,
) -> pd.Series:
    """Recent spread std / lagged baseline std; values >> 1 flag a variance jump."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")
    if baseline_window < 1:
        raise ValueError(f"baseline_window must be >= 1, got {baseline_window!r}")
    s = spread.astype(float)
    recent = s.rolling(window, min_periods=window).std(ddof=1)
    baseline = s.shift(window).rolling(baseline_window, min_periods=baseline_window).std(
        ddof=1
    )
    ratio = recent / baseline
    return ratio.where(baseline > 0)
