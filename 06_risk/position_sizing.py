"""Thin live Monday-morning sizing helpers (no backtest imports).

PIT reminder (S1 equities)
-------------------------
``past_returns`` / ``past_ic`` may only include weeks whose exit (and label)
is known **before** this Monday's open:

- ``mon_open_mon_open``: exclude the week that exits at this Monday open
  (history through entry date ``t-2``).
- ``mon_open_fri_close``: prior week (Fri exit) is usable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.realized_vol import (
    trailing_open_vol_matrix,
)
from risk.signal_conviction import ICScaleConfig, ic_multiplier_from_history
from risk.vol_targeting import VolTargetConfig, compute_gross_leverage

DEFAULT_INV_VOL_WINDOW = 42


def monday_inv_vol_weights(
    scores: pd.Series,
    opens: pd.DataFrame,
    *,
    decision_date: pd.Timestamp,
    n: int = 15,
    window: int = DEFAULT_INV_VOL_WINDOW,
    pit_shift: int = 1,
) -> pd.Series:
    """
    Dollar-neutral inv-vol base book for one Monday: long ``+``, short ``-``.

    Each sleeve renormalizes to ``±0.5``. Apply ``monday_gross_leverage`` scalar
    separately (``w_lev = w * L`` preserves sign).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if window < 1:
        raise ValueError("window must be >= 1")

    # 1. Score cross-section for this decision date
    sc = scores.astype(float).dropna()
    sc.index = pd.Index(sc.index.astype(str).str.strip().str.upper())
    need = 2 * n
    if len(sc) < need:
        return pd.Series(dtype=float)

    # 2. PIT open-to-open trailing vol on decision_date
    dt = pd.Timestamp(decision_date)
    vol = trailing_open_vol_matrix(opens, window=window, pit_shift=pit_shift)
    if dt not in vol.index:
        return pd.Series(dtype=float)
    vol_row = vol.loc[dt].astype(float)
    vol_row.index = pd.Index(vol_row.index.astype(str).str.strip().str.upper())

    # 3. Top-N / bottom-N (drop overlap)
    top = sc.nlargest(n).index
    bot = sc.nsmallest(n).index
    overlap = top.intersection(bot)
    if len(overlap):
        top = top.difference(overlap)
        bot = bot.difference(overlap)
    if len(top) == 0 or len(bot) == 0:
        return pd.Series(dtype=float)

    # 4. Raw 1/σ then sleeve renormalize to ±0.5
    def _sleeve(names: pd.Index, *, signed_sum: float) -> pd.Series:
        v = vol_row.reindex(names).astype(float)
        raw = (1.0 / v).where(np.isfinite(v) & (v > 0))
        w = raw.replace([np.inf, -np.inf], np.nan).dropna()
        w = w.where(w > 0).dropna()
        if w.empty:
            return w
        total = float(w.sum())
        if total <= 0 or not np.isfinite(total):
            return pd.Series(dtype=float)
        return w * (signed_sum / total)

    long_w = _sleeve(top, signed_sum=0.5)
    short_mag = _sleeve(bot, signed_sum=0.5)
    if long_w.empty or short_mag.empty:
        return pd.Series(dtype=float)

    # 5. Signed series: long +, short -
    out = pd.Series(0.0, index=sc.index, dtype=float)
    out.loc[long_w.index] = long_w.to_numpy()
    out.loc[short_mag.index] = (-short_mag).to_numpy()
    return out.replace(0.0, np.nan).dropna()


def monday_gross_leverage(
    past_returns: pd.Series | list[float] | np.ndarray,
    vol_cfg: VolTargetConfig,
    *,
    past_ic: pd.Series | list[float] | np.ndarray | None = None,
    past_n_names: pd.Series | list[float] | np.ndarray | None = None,
    ic_cfg: ICScaleConfig | None = None,
    prev_leverage: float | None = None,
    max_gross: float | None = None,
) -> dict[str, float]:
    """
    One call site for paper/live: returns ``leverage``, ``l_vol``, ``m_ic``.

    Does not import ``backtest``. Apply ``L`` to dollar-neutral base weights.
    """
    m_ic = 1.0
    if (
        ic_cfg is not None
        and ic_cfg.enabled
        and past_ic is not None
        and past_n_names is not None
    ):
        m_ic = ic_multiplier_from_history(past_ic, past_n_names, ic_cfg)
    elif ic_cfg is not None and not ic_cfg.enabled:
        m_ic = 1.0

    if not vol_cfg.enabled and (ic_cfg is None or not ic_cfg.enabled):
        return {"leverage": 1.0, "l_vol": 1.0, "m_ic": 1.0}

    # When vol targeting is off but IC is on, treat L_vol as 1.0
    if not vol_cfg.enabled:
        lo = float(vol_cfg.min_leverage)
        hi = float(vol_cfg.max_leverage if max_gross is None else max_gross)
        lev = float(np.clip(1.0 * m_ic, lo, hi))
        return {"leverage": lev, "l_vol": 1.0, "m_ic": float(m_ic)}

    lev = compute_gross_leverage(
        past_returns,
        vol_cfg,
        ic_multiplier=m_ic,
        prev_leverage=prev_leverage,
        max_gross=max_gross,
    )
    # Recover L_vol for diagnostics (without IC)
    from risk.vol_targeting import leverage_from_history

    l_vol = leverage_from_history(
        past_returns, vol_cfg, prev_leverage=prev_leverage
    )
    return {"leverage": float(lev), "l_vol": float(l_vol), "m_ic": float(m_ic)}
