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
    a: pd.Series,
    b: pd.Series,
    *,
    delta: float = 1e-4,
    obs_var: float = 1.0,
    burn_in: int = 30,
    vol_span: int = 60,
) -> pd.Series:
    """Time-varying correlation via Kalman on EW-standardized products (H-011).

    Standardize ``a`` / ``b`` with a causal EWM std, observe the product
    ``u_t * v_t`` as a noisy measurement of scalar state ``ρ``, and return the
    **posterior** ``ρ̂_t`` (info through ``t``), clipped to ``[-1, 1]``.
    Burn-in rows are NaN. Uses ``_run_kalman`` only — no second KF core.

    Default ``obs_var=1.0`` matches ``Var(u v) ≈ 1`` under independence for
    approximately unit-variance inputs (unlike the hedge filter's ``1e-3``).
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    if obs_var <= 0.0:
        raise ValueError(f"obs_var must be > 0, got {obs_var!r}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in!r}")
    if vol_span < 2:
        raise ValueError(f"vol_span must be >= 2, got {vol_span!r}")
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    if not a.index.equals(b.index):
        raise ValueError("a and b must share an identical index")

    a_s = a.astype(float)
    b_s = b.astype(float)
    sig_a = a_s.ewm(span=vol_span, min_periods=vol_span, adjust=False).std()
    sig_b = b_s.ewm(span=vol_span, min_periods=vol_span, adjust=False).std()
    u = a_s / sig_a
    v = b_s / sig_b
    product = (u * v).to_numpy(dtype=float)

    n = len(product)
    F = np.ones((n, 1), dtype=float)
    q_scale = delta / (1.0 - delta)
    Q = q_scale * np.eye(1)
    theta0 = np.zeros(1, dtype=float)
    P0 = np.eye(1, dtype=float)

    _, theta_post, _, _, _ = _run_kalman(
        product, F, Q, float(obs_var), theta0, P0
    )
    rho = np.clip(theta_post[:, 0], -1.0, 1.0)
    if burn_in > 0:
        rho[: min(burn_in, n)] = np.nan
    # No finite product → no update; keep NaN rather than a stale zero state.
    bad = ~np.isfinite(product)
    rho[bad] = np.nan
    return pd.Series(rho, index=a.index, dtype=float, name="rho")
