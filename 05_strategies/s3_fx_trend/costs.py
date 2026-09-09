"""OANDA S3 FX cost model: spread + slippage + swap (rate-diff + financing)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_COST_PROFILE = "A_FX_OANDA_S3"

COSTS: dict[str, dict[str, Any]] = {
    "A_FX_OANDA_S3": {
        "model": "spread_plus_slippage_plus_swap",
        "pair_spread_pips": {
            "EURUSD": 1.4,
            "USDJPY": 1.4,
            "GBPUSD": 2.0,
            "USDCHF": 1.8,
            "USDCAD": 2.2,
            "AUDUSD": 1.4,
            "NZDUSD": 1.7,
        },
        "slippage_pips_per_leg": 0.1,
        "swap": {
            "model": "rate_diff_plus_financing_spread",
            "financing_spread_bps_annual": 50.0,
            "trading_days_per_year": 365.0,
            "wednesday_triple_swap": True,
        },
        "conversion_markup_bps": 0.0,
    },
}


def _normalize_pair(pair: str) -> str:
    p = str(pair).strip().upper().replace("_", "").replace("/", "").replace("=X", "")
    return p


def fx_pip_size(pair: str) -> float:
    """Pip size: 0.01 for JPY pairs, else 0.0001."""
    p = _normalize_pair(pair)
    return 0.01 if "JPY" in p else 0.0001


def pair_supported(pair: str) -> bool:
    """True when the pair has a locked spread in ``A_FX_OANDA_S3``."""
    cfg = COSTS[DEFAULT_COST_PROFILE]
    return _normalize_pair(pair) in cfg["pair_spread_pips"]


def resolve_cost_profile(pair: str, override: str | None = None) -> str:
    """Return the cost-table key (optional override). Unsupported → default key."""
    if override:
        return str(override)
    p = _normalize_pair(pair)
    if not pair_supported(p):
        logger.warning("pair %s not in A_FX_OANDA_S3 spreads; using default profile", p)
    return DEFAULT_COST_PROFILE


def leg_cost_bps(pair: str, price: float) -> float:
    """Half-spread + slippage in bps of notional for one FX leg."""
    profile = resolve_cost_profile(pair)
    cfg = COSTS[profile]
    p = _normalize_pair(pair)
    pips = float(cfg["pair_spread_pips"].get(p, 1.5))
    pip_size = fx_pip_size(p)
    px = max(float(price), 1e-12)
    spread_bps = (pips * pip_size / px) * 10_000.0
    slip_bps = float(cfg["slippage_pips_per_leg"]) * pip_size / px * 10_000.0
    return 0.5 * spread_bps + slip_bps


def daily_swap_return(
    pair: str,
    weight: float,
    rate_diff: float,
    *,
    financing_spread_bps_annual: float = 50.0,
    wednesday_triple: bool = True,
    asof: pd.Timestamp | str | None = None,
    trading_days_per_year: float = 365.0,
) -> float:
    """Daily swap PnL in return units for a signed pair weight.

    Earns ``rate_diff`` (base minus quote, annual decimal) in the direction of
    the position; financing spread is always a cost on gross ``|weight|``.
    OANDA triples Wednesday accruals when ``wednesday_triple`` is True.
    """
    _ = pair  # reserved for pair-specific schedules
    w = float(weight)
    if w == 0.0 or not np.isfinite(w):
        return 0.0
    rd = float(rate_diff) if np.isfinite(rate_diff) else 0.0
    fin = float(financing_spread_bps_annual) / 10_000.0
    days = float(trading_days_per_year) if trading_days_per_year > 0 else 365.0
    pos_sign = 1.0 if w > 0 else -1.0
    raw = (rd * pos_sign - fin) * abs(w) / days
    if wednesday_triple and asof is not None:
        ts = pd.Timestamp(asof)
        if int(ts.dayofweek) == 2:  # Wednesday
            raw *= 3.0
    return float(raw)
