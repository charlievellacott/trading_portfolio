"""Frozen S2 simulation config (research runner; not a live Strategy subclass)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


VALID_HEDGE = frozenset({"ols", "kalman"})
VALID_BAR = frozenset({"1d", "1h"})
VALID_BREAK = frozenset({"off", "block_05_flat_10", "flat_05"})
VALID_TREND = frozenset({"off", "adx_veto", "rsi_confirm", "both"})
VALID_OVERLAP = frozenset({"allow", "never_allow"})
VALID_EXIT = frozenset({"mean_only", "hl3_atr_breaker"})
VALID_SIZE = frozenset({"equal", "score", "score_conf"})
VALID_VOL = frozenset({"fixed_k", "kt", "s1_vt"})
VALID_Z_WINDOW = frozenset({"fixed", "adaptive", "adaptive_alt"})
VALID_ENTRY = frozenset(
    {"trad_z", "v1_roll_asym", "v1_ewm_asym", "v2_ou", "v3_hmm_innov"}
)


@dataclass(frozen=True)
class S2SimConfig:
    """One-knob research config. Defaults match the H-001 trad-z OLS book."""

    hedge: str = "ols"
    bar: str = "1d"
    entry_z: float = 2.0
    exit_z: float = 0.0
    k_in: float = 2.0
    k_out: float = 0.0
    beta_column: str = "beta"
    score_column: str = "z"
    use_hedge_ratio_sizing: bool = True
    ols_window: int = 252
    z_window: int = 60
    hl_window: int = 252
    break_mode: str = "off"
    trend_mode: str = "off"
    hl_gate_min: float | None = None
    hl_gate_max: float | None = None
    overlap_mode: str = "allow"
    exit_mode: str = "mean_only"
    n_half_lives: float = 3.0
    atr_risk_frac: float = 0.01
    pair_max_loss: float = -0.20
    atr_window: int = 14
    corr_k: float | None = None
    size_mode: str = "equal"
    vol_mode: str = "fixed_k"
    k0: float = 2.0
    sigma_window: int = 60
    sigma_bar: float | None = None
    vt_target_ann_vol: float = 0.10
    z_window_mode: str = "fixed"
    z_clip_min: int = 20
    z_clip_max: int = 120
    entry_mode: str = "trad_z"
    hmm_mr_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.hedge not in VALID_HEDGE:
            raise ValueError(f"hedge must be in {sorted(VALID_HEDGE)}")
        if self.bar not in VALID_BAR:
            raise ValueError(f"bar must be in {sorted(VALID_BAR)} (no 4h for universe C)")
        if self.break_mode not in VALID_BREAK:
            raise ValueError(f"break_mode must be in {sorted(VALID_BREAK)}")
        if self.trend_mode not in VALID_TREND:
            raise ValueError(f"trend_mode must be in {sorted(VALID_TREND)}")
        if self.overlap_mode not in VALID_OVERLAP:
            raise ValueError(f"overlap_mode must be in {sorted(VALID_OVERLAP)}")
        if self.exit_mode not in VALID_EXIT:
            raise ValueError(f"exit_mode must be in {sorted(VALID_EXIT)}")
        if self.size_mode not in VALID_SIZE:
            raise ValueError(f"size_mode must be in {sorted(VALID_SIZE)}")
        if self.vol_mode not in VALID_VOL:
            raise ValueError(f"vol_mode must be in {sorted(VALID_VOL)}")
        if self.z_window_mode not in VALID_Z_WINDOW:
            raise ValueError(f"z_window_mode must be in {sorted(VALID_Z_WINDOW)}")
        if self.entry_mode not in VALID_ENTRY:
            raise ValueError(f"entry_mode must be in {sorted(VALID_ENTRY)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> S2SimConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})
