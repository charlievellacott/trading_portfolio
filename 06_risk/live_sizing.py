"""Thin live Monday-morning gross-leverage helper (no backtest imports).

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

from risk.signal_conviction import ICScaleConfig, ic_multiplier_from_history
from risk.vol_targeting import VolTargetConfig, compute_gross_leverage


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
