"""Shared linear-Gaussian Kalman filter primitives (S2 / reusable)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _run_kalman(
    y: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: float,
    theta0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear-Gaussian filter with time-varying observation matrix F[t].

    Returns prior theta, prior P diagonals unused externally; also e, S, and
    posterior theta after update (caller may ignore posterior for PIT spread).
    """
    n = len(y)
    k = theta0.shape[0]
    theta_prior = np.full((n, k), np.nan)
    e_arr = np.full(n, np.nan)
    S_arr = np.full(n, np.nan)
    theta_post = np.full((n, k), np.nan)

    theta = theta0.astype(float).copy()
    P = P0.astype(float).copy()
    Q = Q.astype(float)
    eye = np.eye(k)

    for t in range(n):
        yt = y[t]
        Ft = F[t]
        if not np.isfinite(yt) or not np.all(np.isfinite(Ft)):
            # Random-walk predict without update when observation missing.
            theta = theta.copy()
            P = P + Q
            continue

        # Predict (random walk: F_state = I)
        theta_pred = theta
        P_pred = P + Q

        theta_prior[t] = theta_pred
        e = float(yt - Ft @ theta_pred)
        S = float(Ft @ P_pred @ Ft.T + R)
        e_arr[t] = e
        S_arr[t] = S

        if S <= 0.0 or not np.isfinite(S):
            theta = theta_pred
            P = P_pred
            theta_post[t] = theta
            continue

        K = (P_pred @ Ft.T) / S
        theta = theta_pred + K * e
        P = (eye - np.outer(K, Ft)) @ P_pred
        theta_post[t] = theta

    return theta_prior, theta_post, e_arr, S_arr, P


def kalman_linear_regression(
    y: pd.Series,
    x: pd.Series,
    *,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
    burn_in: int = 30,
) -> pd.DataFrame:
    """Time-varying OLS via Kalman; spread ``s_t = y_t - alpha_t - beta_t * x_t`` from prior state.

    Returns columns:
    - beta, alpha: hedge known before seeing y_t (prior, not posterior)
    - spread: residual / innovation e_t (prediction error of y_t)
    - spread_var: filter's predicted variance S_t of that residual
    - z_innov: e_t / sqrt(S_t) — same idea as a rolling z, but scaled by the model variance
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    if obs_var <= 0.0:
        raise ValueError(f"obs_var must be > 0, got {obs_var!r}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in!r}")
    if len(y) != len(x):
        raise ValueError("y and x must have the same length")
    if not y.index.equals(x.index):
        raise ValueError("y and x must share an identical index")

    y_arr = y.astype(float).to_numpy()
    x_arr = x.astype(float).to_numpy()
    n = len(y_arr)

    # Observation matrix F_t = [x_t, 1] for state [beta, alpha]
    F = np.column_stack([x_arr, np.ones(n, dtype=float)])
    q_scale = delta / (1.0 - delta)
    Q = q_scale * np.eye(2)
    theta0 = np.zeros(2, dtype=float)
    P0 = np.eye(2, dtype=float)

    theta_prior, _, e_arr, S_arr, _ = _run_kalman(
        y_arr, F, Q, float(obs_var), theta0, P0
    )

    beta = theta_prior[:, 0].copy()
    alpha = theta_prior[:, 1].copy()
    spread = e_arr.copy()
    spread_var = S_arr.copy()
    z_innov = np.full(n, np.nan)
    ok = np.isfinite(spread) & np.isfinite(spread_var) & (spread_var > 0.0)
    z_innov[ok] = spread[ok] / np.sqrt(spread_var[ok])

    if burn_in > 0:
        end = min(burn_in, n)
        beta[:end] = np.nan
        alpha[:end] = np.nan
        spread[:end] = np.nan
        spread_var[:end] = np.nan
        z_innov[:end] = np.nan

    return pd.DataFrame(
        {
            "beta": beta,
            "alpha": alpha,
            "spread": spread,
            "spread_var": spread_var,
            "z_innov": z_innov,
        },
        index=y.index,
    )


def kalman_correlation(
    r1: pd.Series,
    r2: pd.Series,
    *,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
    burn_in: int = 30,
) -> pd.Series:
    """PIT correlation of two series via shared ``_run_kalman`` on second moments.

    Each of ``r1^2``, ``r2^2``, and ``r1 r2`` is tracked as a 1-state random
    walk (prior, not posterior). ``rho_hat = cov / (s1 s2)`` from those priors.
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    if obs_var <= 0.0:
        raise ValueError(f"obs_var must be > 0, got {obs_var!r}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in!r}")
    if len(r1) != len(r2):
        raise ValueError("r1 and r2 must have the same length")
    if not r1.index.equals(r2.index):
        raise ValueError("r1 and r2 must share an identical index")

    a = r1.astype(float).to_numpy()
    b = r2.astype(float).to_numpy()
    n = len(a)
    F = np.ones((n, 1), dtype=float)
    q_scale = delta / (1.0 - delta)
    Q = np.array([[q_scale]], dtype=float)
    theta0 = np.array([0.0], dtype=float)
    P0 = np.array([[1.0]], dtype=float)

    def _moment(obs: np.ndarray) -> np.ndarray:
        prior, _, _, _, _ = _run_kalman(
            obs, F, Q, float(obs_var), theta0, P0
        )
        return prior[:, 0]

    v1 = _moment(a * a)
    v2 = _moment(b * b)
    c12 = _moment(a * b)
    rho = np.full(n, np.nan)
    ok = (
        np.isfinite(v1)
        & np.isfinite(v2)
        & np.isfinite(c12)
        & (v1 > 0.0)
        & (v2 > 0.0)
    )
    denom = np.sqrt(v1[ok] * v2[ok])
    rho[ok] = np.clip(c12[ok] / denom, -1.0, 1.0)
    if burn_in > 0:
        rho[: min(burn_in, n)] = np.nan
    return pd.Series(rho, index=r1.index, dtype=float, name="rho_hat")
