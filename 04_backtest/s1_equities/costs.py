"""Alpaca-representative transaction costs for S1 equities backtests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlpacaCostModel:
    """
    One-way cost components in basis points.

    Alpaca US equities are commission-free; regulatory fees are tiny.
    Dominant costs are half-spread + venue/auction slippage.
    Monday open fills use higher slippage than Friday close fills.
    """

    commission_bps: float = 0.0
    regulatory_bps: float = 0.5
    half_spread_bps: float = 5.0
    open_slippage_bps: float = 10.0
    close_slippage_bps: float = 5.0
    short_borrow_bps: float = 0.0

    @property
    def open_one_way_bps(self) -> float:
        return (
            self.commission_bps
            + self.regulatory_bps
            + self.half_spread_bps
            + self.open_slippage_bps
            + self.short_borrow_bps
        )

    @property
    def close_one_way_bps(self) -> float:
        return (
            self.commission_bps
            + self.regulatory_bps
            + self.half_spread_bps
            + self.close_slippage_bps
            + self.short_borrow_bps
        )

    def one_way_fraction(self, *, side: str) -> float:
        """Return one-way cost as a fraction of traded notional."""
        if side == "open":
            return self.open_one_way_bps / 10_000.0
        if side == "close":
            return self.close_one_way_bps / 10_000.0
        raise ValueError(f"side must be 'open' or 'close', got {side!r}")


DEFAULT_COSTS = AlpacaCostModel()


def cost_summary(model: AlpacaCostModel = DEFAULT_COSTS) -> dict[str, float]:
    """Dict of named cost components for notebook display."""
    return {
        "commission_bps": model.commission_bps,
        "regulatory_bps": model.regulatory_bps,
        "half_spread_bps": model.half_spread_bps,
        "open_slippage_bps": model.open_slippage_bps,
        "close_slippage_bps": model.close_slippage_bps,
        "short_borrow_bps": model.short_borrow_bps,
        "open_one_way_bps": model.open_one_way_bps,
        "close_one_way_bps": model.close_one_way_bps,
    }
