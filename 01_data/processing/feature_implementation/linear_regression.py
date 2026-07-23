"""Reusable rolling OLS regression primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ols_stats(
    y: pd.Series,
    x: pd.Series,
    window: int,
    *,
    include_idio_vol: bool = False,
) -> pd.DataFrame:
    """
    Rolling OLS of ``y`` on ``x`` ending at each bar ``t``.

    Uses the last ``window`` jointly finite ``(y, x)`` pairs with
    ``min_periods=window``. ``Var(x)==0`` or ``SS_tot==0`` -> NaN outputs.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    n = len(y)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    idio_vol = np.full(n, np.nan) if include_idio_vol else None

    y_arr = y.to_numpy(dtype=float)
    x_arr = x.to_numpy(dtype=float)

    for i in range(window - 1, n):
        y_w = y_arr[i - window + 1 : i + 1]
        x_w = x_arr[i - window + 1 : i + 1]
        valid = np.isfinite(y_w) & np.isfinite(x_w)
        if int(valid.sum()) < window:
            continue
        y_v = y_w[valid]
        x_v = x_w[valid]
        x_demean = x_v - x_v.mean()
        ss_xx = np.sum(x_demean ** 2)
        if ss_xx == 0.0:
            continue
        y_demean = y_v - y_v.mean()
        b = np.sum(x_demean * y_demean) / ss_xx
        a = y_v.mean() - b * x_v.mean()
        eps = y_v - a - b * x_v
        ss_res = np.sum(eps ** 2)
        ss_tot = np.sum(y_demean ** 2)
        alpha[i] = a
        beta[i] = b
        r2[i] = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        if include_idio_vol and idio_vol is not None:
            idio_vol[i] = float(np.std(eps, ddof=1))

    out: dict[str, np.ndarray] = {"alpha": alpha, "beta": beta, "r2": r2}
    if include_idio_vol and idio_vol is not None:
        out["idio_vol"] = idio_vol
    return pd.DataFrame(out, index=y.index)


def rolling_residual(
    y: pd.Series,
    x: pd.Series,
    alpha: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """Per-bar residual ``y - alpha - beta * x`` using aligned rolling coefficients."""
    return y.astype(float) - alpha.astype(float) - beta.astype(float) * x.astype(float)


def rolling_conditional_ols_stats(
    y: pd.Series,
    x: pd.Series,
    window: int,
    *,
    side: str,
    min_obs: int | None = None,
) -> pd.DataFrame:
    """
    Rolling OLS of ``y`` on ``x`` restricted to bars where ``x`` is below
    (``side='down'``) or at/above (``side='up'``) the in-window mean of ``x``.

    Parameters
    ----------
    side : {'down', 'up'}
    min_obs : int or None
        Minimum observations in the conditional subset for the regression to be
        valid. Default ``max(20, window // 4)``.

    Returns DataFrame with columns ``alpha``, ``beta``, ``r2``, ``n_obs``.
    """
    if side not in ("down", "up"):
        raise ValueError(f"side must be 'down' or 'up', got {side!r}")
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_obs is None:
        min_obs = max(20, window // 4)
    if min_obs < 2:
        min_obs = 2

    n = len(y)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    n_obs_arr = np.full(n, np.nan)

    y_arr = y.to_numpy(dtype=float)
    x_arr = x.to_numpy(dtype=float)

    for i in range(window - 1, n):
        y_w = y_arr[i - window + 1 : i + 1]
        x_w = x_arr[i - window + 1 : i + 1]
        valid = np.isfinite(y_w) & np.isfinite(x_w)
        if int(valid.sum()) < window:
            continue
        y_v = y_w[valid]
        x_v = x_w[valid]

        x_mean = x_v.mean()
        if side == "down":
            mask = x_v < x_mean
        else:
            mask = x_v >= x_mean

        count = int(mask.sum())
        if count < min_obs:
            continue

        y_sub = y_v[mask]
        x_sub = x_v[mask]
        x_demean = x_sub - x_sub.mean()
        ss_xx = np.sum(x_demean ** 2)
        if ss_xx == 0.0:
            continue

        y_demean = y_sub - y_sub.mean()
        b = np.sum(x_demean * y_demean) / ss_xx
        a = y_sub.mean() - b * x_sub.mean()
        eps = y_sub - a - b * x_sub
        ss_res = np.sum(eps ** 2)
        ss_tot = np.sum(y_demean ** 2)

        alpha[i] = a
        beta[i] = b
        r2[i] = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        n_obs_arr[i] = count

    return pd.DataFrame(
        {"alpha": alpha, "beta": beta, "r2": r2, "n_obs": n_obs_arr},
        index=y.index,
    )


def rolling_multi_ols_stats(
    y: pd.Series,
    X: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Rolling multivariate OLS of ``y`` on columns of ``X`` with intercept.

    Uses the last ``window`` jointly finite rows. NaN when the design matrix
    is rank-deficient or fewer than ``window`` valid observations exist.

    Returns DataFrame with columns ``["alpha", *X.columns]`` (intercept + slopes).
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    regressors = list(X.columns)
    k = len(regressors)
    n = len(y)

    out_alpha = np.full(n, np.nan)
    out_slopes = np.full((n, k), np.nan)

    y_arr = y.to_numpy(dtype=float)
    X_arr = X.to_numpy(dtype=float)

    for i in range(window - 1, n):
        y_w = y_arr[i - window + 1 : i + 1]
        X_w = X_arr[i - window + 1 : i + 1]
        valid = np.isfinite(y_w) & np.all(np.isfinite(X_w), axis=1)
        if int(valid.sum()) < window:
            continue

        y_v = y_w[valid]
        X_v = X_w[valid]
        ones = np.ones((len(y_v), 1))
        design = np.hstack([ones, X_v])

        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue

        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue

        out_alpha[i] = coeffs[0]
        out_slopes[i] = coeffs[1:]

    result: dict[str, np.ndarray] = {"alpha": out_alpha}
    for j, col in enumerate(regressors):
        result[col] = out_slopes[:, j]
    return pd.DataFrame(result, index=y.index)
