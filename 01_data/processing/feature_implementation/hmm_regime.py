"""H-013 2-state Gaussian HMM on spread changes (PIT filter, not a Kalman)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GaussianHMM2Params:
    """Two-state univariate Gaussian HMM parameters (fit on fold-train only)."""

    means: tuple[float, float]
    vars: tuple[float, float]
    trans: tuple[tuple[float, float], tuple[float, float]]
    start: tuple[float, float]
    mr_state: int


def _log_norm(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    v = max(float(var), 1e-12)
    return -0.5 * (np.log(2.0 * np.pi * v) + (x - mean) ** 2 / v)


def fit_gaussian_hmm_2state(
    x: pd.Series | np.ndarray,
    *,
    n_iter: int = 40,
    seed: int = 0,
) -> GaussianHMM2Params:
    """EM fit of a 2-state Gaussian HMM. ``mr_state`` is the state with mean closer to 0."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        raise ValueError("need at least 20 finite observations to fit HMM")

    rng = np.random.default_rng(seed)
    lo, hi = np.quantile(arr, [0.25, 0.75])
    means = np.array([float(lo), float(hi)], dtype=float)
    vars_ = np.array([float(np.var(arr)) + 1e-8] * 2, dtype=float)
    trans = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    start = np.array([0.5, 0.5], dtype=float)
    # jitter so the two means are not identical
    means = means + rng.normal(0.0, 1e-6, size=2)

    n = len(arr)
    for _ in range(n_iter):
        ll = np.column_stack(
            [_log_norm(arr, means[0], vars_[0]), _log_norm(arr, means[1], vars_[1])]
        )
        log_start = np.log(np.clip(start, 1e-12, 1.0))
        log_trans = np.log(np.clip(trans, 1e-12, 1.0))
        alpha = np.full((n, 2), -np.inf)
        alpha[0] = log_start + ll[0]
        for t in range(1, n):
            for j in range(2):
                alpha[t, j] = ll[t, j] + np.logaddexp(
                    alpha[t - 1, 0] + log_trans[0, j],
                    alpha[t - 1, 1] + log_trans[1, j],
                )
        beta = np.zeros((n, 2))
        for t in range(n - 2, -1, -1):
            for i in range(2):
                beta[t, i] = np.logaddexp(
                    log_trans[i, 0] + ll[t + 1, 0] + beta[t + 1, 0],
                    log_trans[i, 1] + ll[t + 1, 1] + beta[t + 1, 1],
                )
        gamma_log = alpha + beta
        gamma_log -= np.max(gamma_log, axis=1, keepdims=True)
        gamma = np.exp(gamma_log)
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((n - 1, 2, 2))
        for t in range(n - 1):
            m = np.full((2, 2), -np.inf)
            for i in range(2):
                for j in range(2):
                    m[i, j] = (
                        alpha[t, i]
                        + log_trans[i, j]
                        + ll[t + 1, j]
                        + beta[t + 1, j]
                    )
            m -= np.max(m)
            e = np.exp(m)
            xi[t] = e / e.sum()

        start = gamma[0]
        trans = xi.sum(axis=0)
        trans = trans / np.clip(trans.sum(axis=1, keepdims=True), 1e-12, None)
        for j in range(2):
            w = gamma[:, j]
            sw = float(w.sum())
            if sw <= 1e-12:
                continue
            means[j] = float((w * arr).sum() / sw)
            vars_[j] = float((w * (arr - means[j]) ** 2).sum() / sw) + 1e-8

    mr_state = int(np.argmin(np.abs(means)))
    return GaussianHMM2Params(
        means=(float(means[0]), float(means[1])),
        vars=(float(vars_[0]), float(vars_[1])),
        trans=(
            (float(trans[0, 0]), float(trans[0, 1])),
            (float(trans[1, 0]), float(trans[1, 1])),
        ),
        start=(float(start[0]), float(start[1])),
        mr_state=mr_state,
    )


def filter_mr_probability(
    x: pd.Series,
    params: GaussianHMM2Params,
) -> pd.Series:
    """Forward-filter P(MR state | x_1:t). Does not use future observations."""
    arr = x.astype(float).to_numpy()
    n = len(arr)
    means = np.array(params.means, dtype=float)
    vars_ = np.array(params.vars, dtype=float)
    log_trans = np.log(np.clip(np.array(params.trans, dtype=float), 1e-12, 1.0))
    log_start = np.log(np.clip(np.array(params.start, dtype=float), 1e-12, 1.0))
    out = np.full(n, np.nan)
    log_alpha = log_start.copy()
    for t in range(n):
        xt = arr[t]
        if not np.isfinite(xt):
            continue
        ll = np.array(
            [_log_norm(np.array([xt]), means[0], vars_[0])[0],
             _log_norm(np.array([xt]), means[1], vars_[1])[0]],
            dtype=float,
        )
        if t == 0:
            log_alpha = log_start + ll
        else:
            nxt = np.empty(2)
            for j in range(2):
                nxt[j] = ll[j] + np.logaddexp(
                    log_alpha[0] + log_trans[0, j],
                    log_alpha[1] + log_trans[1, j],
                )
            log_alpha = nxt
        m = float(np.max(log_alpha))
        p = np.exp(log_alpha - m)
        p = p / p.sum()
        out[t] = float(p[int(params.mr_state)])
    return pd.Series(out, index=x.index, dtype=float, name="p_mr")
