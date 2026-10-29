"""S3 cost models: OANDA FX (spread + swap) and Alpaca ETF (fees + slippage)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from data.processing.s3_price_panel import ETF_MULTI_TICKERS

logger = logging.getLogger(__name__)

FX_COST_PROFILE = "A_FX_OANDA_S3"
ETF_COST_PROFILE = "B_ETF_ALPACA"
DEFAULT_COST_PROFILE = FX_COST_PROFILE
PROFILE_BY_UNIVERSE = {
    "A": FX_COST_PROFILE,
    "B": ETF_COST_PROFILE,
}

COSTS: dict[str, dict[str, Any]] = {
    FX_COST_PROFILE: {
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
    ETF_COST_PROFILE: {
        "model": "percent_commission_plus_slippage",
        "commission_bps": 0.0,
        "third_party_fee_bps": 0.1,
        "slippage_bps_per_leg": 3.2,
        "borrow_bps_annual": 0.0,
        "trading_days_per_year": 252.0,
        "commission_ccy": "USD",
    },
}


def _normalize_pair(pair: str) -> str:
    p = str(pair).strip().upper().replace("_", "").replace("/", "").replace("=X", "")
    return p


_ETF_TICKERS = frozenset(_normalize_pair(t) for t in ETF_MULTI_TICKERS)


def fx_pip_size(pair: str) -> float:
    """Pip size: 0.01 for JPY pairs, else 0.0001."""
    p = _normalize_pair(pair)
    return 0.01 if "JPY" in p else 0.0001


def cost_profile_for_universe(universe: str | None) -> str:
    """Core universe A → OANDA; B → Alpaca ETF. Missing / event → OANDA."""
    u = str(universe or "").strip().upper()
    return PROFILE_BY_UNIVERSE.get(u, FX_COST_PROFILE)


def pair_supported(pair: str, profile: str | None = None) -> bool:
    """True when ``pair`` is in the locked table for ``profile`` (default FX)."""
    p = _normalize_pair(pair)
    key = str(profile or DEFAULT_COST_PROFILE)
    if key == ETF_COST_PROFILE:
        return p in _ETF_TICKERS
    cfg = COSTS[FX_COST_PROFILE]
    return p in cfg["pair_spread_pips"]


def resolve_cost_profile(
    pair: str,
    override: str | None = None,
    universe: str | None = None,
) -> str:
    """Return the cost-table key. Override wins, then universe, then instrument."""
    if override:
        return str(override)
    u = str(universe or "").strip().upper()
    if u in PROFILE_BY_UNIVERSE:
        return PROFILE_BY_UNIVERSE[u]
    p = _normalize_pair(pair)
    if p in _ETF_TICKERS:
        return ETF_COST_PROFILE
    if not pair_supported(p, FX_COST_PROFILE):
        logger.warning("pair %s not in A_FX_OANDA_S3 spreads; using default profile", p)
    return DEFAULT_COST_PROFILE


def _fx_leg_cost_bps(pair: str, price: float, cfg: dict) -> float:
    p = _normalize_pair(pair)
    pips = float(cfg["pair_spread_pips"].get(p, 1.5))
    pip_size = fx_pip_size(p)
    px = max(float(price), 1e-12)
    spread_bps = (pips * pip_size / px) * 10_000.0
    slip_bps = float(cfg["slippage_pips_per_leg"]) * pip_size / px * 10_000.0
    return 0.5 * spread_bps + slip_bps


def _etf_leg_cost_bps(cfg: dict) -> float:
    return (
        float(cfg.get("commission_bps", 0.0))
        + float(cfg.get("third_party_fee_bps", 0.0))
        + float(cfg.get("slippage_bps_per_leg", 0.0))
    )


def leg_cost_bps(
    pair: str,
    price: float,
    *,
    profile: str | None = None,
    universe: str | None = None,
) -> float:
    """One-leg cost in bps of notional (FX half-spread+slip, or ETF fees+slip)."""
    key = resolve_cost_profile(pair, override=profile, universe=universe)
    cfg = COSTS[key]
    if cfg["model"] == "percent_commission_plus_slippage":
        return _etf_leg_cost_bps(cfg)
    return _fx_leg_cost_bps(pair, price, cfg)


def daily_borrow_return(
    profile: str,
    *,
    weight: float,
    scale: float = 1.0,
) -> float:
    """Pro-rated daily borrow drag on a short weight (return units, not bps)."""
    cfg = COSTS.get(profile)
    if not cfg or "borrow_bps_annual" not in cfg:
        return 0.0
    short_gross = max(0.0, -float(weight))
    if short_gross <= 0:
        return 0.0
    days = float(cfg.get("trading_days_per_year", 252.0))
    if days <= 0:
        return 0.0
    bps_daily = float(cfg["borrow_bps_annual"]) / days
    return -(bps_daily / 10_000.0) * short_gross * float(scale)


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
