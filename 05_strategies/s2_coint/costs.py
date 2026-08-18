"""H-001 baseline market routing and per-leg transaction costs (no stress overlay)."""

from __future__ import annotations

# Per-leg target notional used for minimum-fee conversions.
LEG_NOTIONAL_LOCAL: dict[str, float] = {
    "A_FX_OANDA": 100_000.0,
    "B_CRYPTO_KRAKEN": 10_000.0,
    "C_HK_IBKR": 100_000.0,
    "C_JP_IBKR": 1_000_000.0,
}

COSTS: dict[str, dict] = {
    "A_FX_OANDA": {
        "model": "spread_plus_slippage",
        "pair_spread_pips": {
            "AUDUSD=X": 1.1,
            "NZDUSD=X": 1.4,
            "EURUSD=X": 1.0,
            "GBPUSD=X": 1.2,
            "USDCHF=X": 1.3,
            "USDJPY=X": 1.1,
        },
        "slippage_pips_per_leg": 0.1,
    },
    "B_CRYPTO_KRAKEN": {
        "model": "maker_taker_plus_slippage",
        "maker_fee_bps": 22.0,
        "taker_fee_bps": 38.0,
        "assume_taker_for_baseline": True,
        "slippage_bps_per_leg": 5.0,
    },
    "C_HK_IBKR": {
        "model": "percent_commission_plus_min_plus_slippage",
        "commission_bps": 8.0,
        "commission_min_local": 18.0,
        "commission_ccy": "HKD",
        "third_party_fee_bps": 13.0,
        "slippage_bps_per_leg": 8.0,
    },
    "C_JP_IBKR": {
        "model": "percent_commission_plus_min_plus_slippage",
        "commission_bps": 8.0,
        "commission_min_local": 80.0,
        "commission_ccy": "JPY",
        "third_party_fee_bps": 2.0,
        "slippage_bps_per_leg": 6.0,
    },
}


def market_profile_for_ticker(ticker: str) -> str:
    if ticker.endswith("=X"):
        return "A_FX_OANDA"
    if ticker.upper().endswith("-USD"):
        return "B_CRYPTO_KRAKEN"
    if ticker.endswith(".HK"):
        return "C_HK_IBKR"
    if ticker.endswith(".T"):
        return "C_JP_IBKR"
    raise ValueError(f"unknown market profile for ticker {ticker}")


def market_profile_for_pair(pair_id: str, ticker_y: str, ticker_x: str) -> str:
    py = market_profile_for_ticker(ticker_y)
    px = market_profile_for_ticker(ticker_x)
    if py != px:
        raise ValueError(
            f"{pair_id}: mixed profiles not supported in H-001 baseline ({py} vs {px})"
        )
    return py


def fx_pip_size(ticker: str) -> float:
    return 0.01 if ticker.startswith("USDJPY") else 0.0001


def fx_leg_cost_bps(ticker: str, price: float, cfg: dict) -> float:
    pips = float(cfg["pair_spread_pips"].get(ticker, 1.2))
    pip_size = fx_pip_size(ticker)
    px = max(float(price), 1e-12)
    spread_bps = (pips * pip_size / px) * 10_000.0
    slip_bps = float(cfg["slippage_pips_per_leg"]) * pip_size / px * 10_000.0
    return 0.5 * spread_bps + slip_bps


def percent_min_bps(cfg: dict, profile: str) -> float:
    bps = (
        float(cfg["commission_bps"])
        + float(cfg["third_party_fee_bps"])
        + float(cfg["slippage_bps_per_leg"])
    )
    min_local = float(cfg["commission_min_local"])
    notional = float(LEG_NOTIONAL_LOCAL[profile])
    min_bps = (min_local / max(notional, 1e-12)) * 10_000.0
    return max(bps, min_bps)


def leg_cost_bps(profile: str, ticker: str, price: float) -> float:
    cfg = COSTS[profile]
    model = cfg["model"]

    if model == "spread_plus_slippage":
        return fx_leg_cost_bps(ticker, price, cfg)

    if model == "maker_taker_plus_slippage":
        fee = float(
            cfg["taker_fee_bps"] if cfg["assume_taker_for_baseline"] else cfg["maker_fee_bps"]
        )
        return fee + float(cfg["slippage_bps_per_leg"])

    if model == "percent_commission_plus_min_plus_slippage":
        return percent_min_bps(cfg, profile)

    raise ValueError(f"unsupported cost model: {model}")
