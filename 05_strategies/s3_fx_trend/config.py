"""Frozen S3 simulation config (research runner; not a live Strategy subclass)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

try:
    from data.ingestion.fx_fetcher import G10_V1_PAIRS
except ImportError:  # fx_fetcher may land on a sibling branch
    G10_V1_PAIRS = (
        "EURUSD",
        "USDJPY",
        "GBPUSD",
        "USDCHF",
        "USDCAD",
        "AUDUSD",
        "NZDUSD",
    )

VALID_BAR = frozenset({"1d", "1h"})
VALID_SIGNAL_FAMILY = frozenset({"tsmom", "donchian", "both"})
VALID_TSMOM_BLEND = frozenset(
    {"single_12m", "single_3m", "single_1m", "equal_blend"}
)
VALID_DONCHIAN_HL = frozenset(
    {"daily_ny_close", "overlap_confirmed", "ex_asia_eurgbp"}
)
VALID_CONVICTION = frozenset({"sign", "abs"})
VALID_CARRY_MODE = frozenset({"off", "gate", "tilt"})
VALID_VALUE_MODE = frozenset({"off", "gate", "tilt"})
VALID_CARRY_REFRESH = frozenset({"weekly", "monthly"})
VALID_VALUE_REFRESH = frozenset({"monthly", "quarterly"})
VALID_SPIKE_MODE = frozenset({"off", "pair_delever", "book_delever", "both"})
VALID_CORR_CONFLICT = frozenset({"off", "corr_gate", "currency_cap", "both"})
VALID_EVENT_BAR = frozenset({"1d", "1h"})


@dataclass(frozen=True)
class S3SimConfig:
    """One-knob research config for the S3 FX trend sleeve."""

    pairs: tuple[str, ...] = G10_V1_PAIRS
    bar: str = "1d"
    day_boundary_tz: str = "America/New_York"
    day_boundary_hour: int = 17

    signal_family: str = "tsmom"
    tsmom_lookbacks: tuple[int, ...] = (21, 63, 252)
    tsmom_blend: str = "single_12m"
    donchian_n: int = 55
    donchian_hl_source: str = "daily_ny_close"

    pair_vt: bool = False
    book_vt: bool = False
    vt_target_ann_vol: float = 0.10
    conviction_scaling: str = "sign"
    weak_signal_flat_z: float = 0.0
    rebalance_band: float = 0.0

    carry_mode: str = "off"
    carry_refresh: str = "weekly"
    value_mode: str = "off"
    value_refresh: str = "monthly"
    carry_agree_scale: float = 1.25
    carry_disagree_scale: float = 0.75
    value_agree_scale: float = 1.25
    value_disagree_scale: float = 0.75

    spike_mode: str = "off"
    spike_threshold_sigmas: float = 2.0

    corr_conflict_mode: str = "off"
    corr_gate_threshold: float = 0.85
    currency_cap_gross_pct: float = 0.50

    cost_profile: str = "A_FX_OANDA_S3"
    include_swap: bool = True
    financing_spread_bps_annual: float = 50.0

    event_z_gate: float = 1.0
    event_impact_min: int = 2
    event_hold_bars: int = 1
    event_bar: str = "1d"
    surprise_sizing: bool = False

    def __post_init__(self) -> None:
        if self.bar not in VALID_BAR:
            raise ValueError(f"bar must be in {sorted(VALID_BAR)}")
        if self.signal_family not in VALID_SIGNAL_FAMILY:
            raise ValueError(
                f"signal_family must be in {sorted(VALID_SIGNAL_FAMILY)}"
            )
        if self.tsmom_blend not in VALID_TSMOM_BLEND:
            raise ValueError(f"tsmom_blend must be in {sorted(VALID_TSMOM_BLEND)}")
        if self.donchian_hl_source not in VALID_DONCHIAN_HL:
            raise ValueError(
                f"donchian_hl_source must be in {sorted(VALID_DONCHIAN_HL)}"
            )
        if self.conviction_scaling not in VALID_CONVICTION:
            raise ValueError(
                f"conviction_scaling must be in {sorted(VALID_CONVICTION)}"
            )
        if self.carry_mode not in VALID_CARRY_MODE:
            raise ValueError(f"carry_mode must be in {sorted(VALID_CARRY_MODE)}")
        if self.value_mode not in VALID_VALUE_MODE:
            raise ValueError(f"value_mode must be in {sorted(VALID_VALUE_MODE)}")
        if self.carry_refresh not in VALID_CARRY_REFRESH:
            raise ValueError(
                f"carry_refresh must be in {sorted(VALID_CARRY_REFRESH)}"
            )
        if self.value_refresh not in VALID_VALUE_REFRESH:
            raise ValueError(
                f"value_refresh must be in {sorted(VALID_VALUE_REFRESH)}"
            )
        if self.spike_mode not in VALID_SPIKE_MODE:
            raise ValueError(f"spike_mode must be in {sorted(VALID_SPIKE_MODE)}")
        if self.corr_conflict_mode not in VALID_CORR_CONFLICT:
            raise ValueError(
                f"corr_conflict_mode must be in {sorted(VALID_CORR_CONFLICT)}"
            )
        if self.event_bar not in VALID_EVENT_BAR:
            raise ValueError(f"event_bar must be in {sorted(VALID_EVENT_BAR)}")
        if int(self.day_boundary_hour) < 0 or int(self.day_boundary_hour) > 23:
            raise ValueError("day_boundary_hour must be in [0, 23]")
        if not self.tsmom_lookbacks:
            raise ValueError("tsmom_lookbacks must be non-empty")
        if any(int(x) < 1 for x in self.tsmom_lookbacks):
            raise ValueError("tsmom_lookbacks entries must be >= 1")
        if int(self.donchian_n) < 2:
            raise ValueError("donchian_n must be >= 2")
        if float(self.vt_target_ann_vol) <= 0:
            raise ValueError("vt_target_ann_vol must be positive")
        if float(self.currency_cap_gross_pct) <= 0:
            raise ValueError("currency_cap_gross_pct must be positive")
        if int(self.event_hold_bars) < 1:
            raise ValueError("event_hold_bars must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> S3SimConfig:
        names = {f.name for f in fields(cls)}
        cleaned = {k: v for k, v in raw.items() if k in names}
        if "pairs" in cleaned and cleaned["pairs"] is not None:
            cleaned["pairs"] = tuple(cleaned["pairs"])
        if "tsmom_lookbacks" in cleaned and cleaned["tsmom_lookbacks"] is not None:
            cleaned["tsmom_lookbacks"] = tuple(int(x) for x in cleaned["tsmom_lookbacks"])
        return cls(**cleaned)
