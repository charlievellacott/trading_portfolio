"""H-010 pair scale and H-008 ATR risk-unit multiplier."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pair_scale_from_score(
    score: float,
    adf_pvalue: float,
    *,
    size_mode: str,
    mean_abs_score: float = 1.0,
) -> float:
    """Scale so fold-train mean is ~1 when ``size_mode`` uses score."""
    if size_mode == "equal":
        return 1.0
    s = abs(float(score)) if np.isfinite(score) else 0.0
    if size_mode == "score_conf":
        p = float(adf_pvalue) if np.isfinite(adf_pvalue) else 1.0
        p = min(max(p, 0.0), 1.0)
        s = s * (1.0 - p)
    denom = float(mean_abs_score) if np.isfinite(mean_abs_score) and mean_abs_score > 0 else 1.0
    return s / denom


def rolling_mean_abs_score(score: pd.Series, window: int) -> pd.Series:
    """PIT rolling mean of ``|score|`` with ``min_periods=window`` (NaN until full)."""
    w = int(window)
    if w < 1:
        raise ValueError("window must be >= 1")
    return score.astype(float).abs().rolling(w, min_periods=w).mean()


def gross_normalized_legs(side: int, beta: float) -> tuple[float, float]:
    """Dollar weights ``(y_w, x_w)`` with ``|y_w| + |x_w| = 1`` for a spread side."""
    y_w = float(side)
    x_w = -float(side) * float(beta)
    gross = abs(y_w) + abs(x_w)
    if gross > 0:
        y_w /= gross
        x_w /= gross
    return y_w, x_w


def atr_size_multiplier(
    *,
    atr: float,
    beta: float,
    n_pairs: int,
    pair_scale: float,
    leverage: float,
    risk_frac: float = 0.01,
) -> float:
    """Size default-trade ATR stop to ``risk_frac`` of book, then × scale × L.

    Pair returns are later averaged across ``n_pairs``, so the raw pair loss at
    1 ATR (``atr / (1+|beta|)``) is scaled so the book contribution is
    ``risk_frac * pair_scale * leverage``.
    """
    if not np.isfinite(atr) or atr <= 0.0:
        return 1.0
    gross = 1.0 + abs(float(beta))
    raw_loss = float(atr) / max(gross, 1e-12)
    if raw_loss <= 0.0:
        return 1.0
    n = max(int(n_pairs), 1)
    target = float(risk_frac) * float(pair_scale) * float(leverage) * float(n)
    return float(target / raw_loss)
