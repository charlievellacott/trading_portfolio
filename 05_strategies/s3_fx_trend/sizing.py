"""S3 pair / book sizing helpers (inverse-vol, conviction, currency caps)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from risk.analytics.s1_equities.vol_targeting import (
    ESTIMATOR_ROLLING,
    VolTargetConfig,
    leverage_from_history,
    leverage_series,
)


def _rolling_vt_cfg(
    target_ann_vol: float,
    *,
    window: int = 60,
    min_periods: int = 20,
) -> VolTargetConfig:
    return VolTargetConfig(
        enabled=True,
        target_ann_vol=float(target_ann_vol),
        estimator=ESTIMATOR_ROLLING,
        window=int(window),
        periods_per_year=252.0,
        min_periods=int(min_periods),
        min_leverage=0.0,
        max_leverage=5.0,
        warmup_leverage=1.0,
    )


def pair_inverse_vol_weights(
    pair_returns: pd.DataFrame | dict[str, pd.Series],
    *,
    target_ann_vol: float = 0.10,
    window: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    """PIT inverse-vol leverages per pair (columns), via rolling ``VolTargetConfig``.

    Row ``i`` uses only returns strictly before ``i`` (``leverage_series`` contract).
    """
    if isinstance(pair_returns, dict):
        frame = pd.DataFrame(pair_returns)
    else:
        frame = pair_returns.copy()
    frame = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    cfg = _rolling_vt_cfg(
        target_ann_vol, window=window, min_periods=min_periods
    )
    cols: dict[str, pd.Series] = {}
    for col in frame.columns:
        lev = leverage_series(frame[col], cfg)["leverage"]
        cols[str(col)] = lev
    if not cols:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(cols, index=frame.index)


def book_vol_target_leverage(
    book_returns_history: pd.Series | list[float] | np.ndarray,
    target_ann_vol: float = 0.10,
    *,
    window: int = 60,
    min_periods: int = 20,
    prev_leverage: float | None = None,
) -> float:
    """Scalar book leverage from completed PIT-safe book returns."""
    cfg = _rolling_vt_cfg(
        target_ann_vol, window=window, min_periods=min_periods
    )
    return float(
        leverage_from_history(
            book_returns_history, cfg, prev_leverage=prev_leverage
        )
    )


def apply_conviction(signal: pd.Series | float, mode: str) -> pd.Series | float:
    """``sign`` → ±1 (0 stays 0); ``abs`` → clip magnitude into [-1, 1]."""
    mode_l = str(mode).lower()
    if isinstance(signal, pd.Series):
        s = pd.to_numeric(signal, errors="coerce").astype(float)
        if mode_l == "sign":
            out = np.sign(s)
            out = out.where(s != 0.0, 0.0)
            return out.astype(float)
        if mode_l == "abs":
            return s.clip(-1.0, 1.0)
        raise ValueError(f"unknown conviction mode {mode!r}")
    x = float(signal) if np.isfinite(signal) else 0.0
    if mode_l == "sign":
        return float(np.sign(x))
    if mode_l == "abs":
        return float(np.clip(x, -1.0, 1.0))
    raise ValueError(f"unknown conviction mode {mode!r}")


def apply_rebalance_band(
    current_w: float,
    target_w: float,
    band: float,
) -> float:
    """Keep ``current_w`` unless ``|target - current|`` exceeds ``band``."""
    cur = float(current_w)
    tgt = float(target_w)
    b = float(band)
    if b <= 0 or not np.isfinite(b):
        return tgt
    if abs(tgt - cur) <= b:
        return cur
    return tgt


def apply_weak_signal_flat(
    signal: pd.Series | float,
    threshold: float,
) -> pd.Series | float:
    """Zero the signal when ``|signal| < threshold``."""
    thr = float(threshold)
    if thr <= 0:
        return signal
    if isinstance(signal, pd.Series):
        s = pd.to_numeric(signal, errors="coerce").astype(float)
        return s.where(s.abs() >= thr, 0.0)
    x = float(signal) if np.isfinite(signal) else 0.0
    return 0.0 if abs(x) < thr else x


def _normalize_pair(pair: str) -> str:
    return str(pair).strip().upper().replace("_", "").replace("/", "").replace("=X", "")


def currency_gross_exposure(
    weights: pd.Series | dict[str, float],
    pairs: list[str] | tuple[str, ...] | None = None,
) -> pd.Series:
    """Per-currency gross exposure from signed pair weights (base +, quote −)."""
    if isinstance(weights, dict):
        w = pd.Series(weights, dtype=float)
    else:
        w = pd.to_numeric(weights, errors="coerce").astype(float)
    if pairs is not None:
        w = w.reindex([_normalize_pair(p) for p in pairs]).fillna(0.0)
    net: dict[str, float] = defaultdict(float)
    for pair, val in w.items():
        if not np.isfinite(val) or val == 0.0:
            continue
        p = _normalize_pair(str(pair))
        if len(p) != 6:
            continue
        net[p[:3]] += float(val)
        net[p[3:]] -= float(val)
    if not net:
        return pd.Series(dtype=float, name="gross")
    return pd.Series({c: abs(v) for c, v in net.items()}, dtype=float, name="gross")


def enforce_currency_cap(
    target_weights: pd.Series | dict[str, float],
    cap_pct: float,
    *,
    max_iter: int = 50,
) -> pd.Series:
    """Iteratively shrink the marginal offending pair until all gross ≤ ``cap_pct``."""
    if isinstance(target_weights, dict):
        w = pd.Series(target_weights, dtype=float).copy()
    else:
        w = pd.to_numeric(target_weights, errors="coerce").astype(float).copy()
    w.index = [_normalize_pair(str(i)) for i in w.index]
    cap = float(cap_pct)
    if cap <= 0 or w.empty:
        return w

    for _ in range(int(max_iter)):
        gross = currency_gross_exposure(w)
        if gross.empty or float(gross.max()) <= cap + 1e-12:
            break
        offenders = gross[gross > cap]
        # Shrink pairs that touch the worst currency.
        worst_ccy = str(offenders.idxmax())
        candidates: list[str] = []
        for pair, val in w.items():
            if not np.isfinite(val) or abs(val) < 1e-15:
                continue
            p = str(pair)
            if p[:3] == worst_ccy or p[3:] == worst_ccy:
                candidates.append(p)
        if not candidates:
            break
        # Shrink the largest |weight| contributor.
        marg = max(candidates, key=lambda p: abs(float(w.loc[p])))
        scale = cap / max(float(gross.loc[worst_ccy]), 1e-12)
        scale = min(max(scale, 0.0), 1.0)
        w.loc[marg] = float(w.loc[marg]) * scale
    return w
