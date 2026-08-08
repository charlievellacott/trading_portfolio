# imports
import numpy as np


# subroutines
def pct_stop_price(entry_px: float, *, is_long: bool, pct: float) -> float:
    # 1. Validate
    if not np.isfinite(entry_px) or entry_px <= 0:
        raise ValueError(f"entry_px must be positive, got {entry_px!r}")
    if not np.isfinite(pct) or pct <= 0:
        raise ValueError(f"pct must be positive, got {pct!r}")

    # 2. Adverse move
    frac = float(pct) / 100.0
    if is_long:
        return float(entry_px) * (1.0 - frac)
    return float(entry_px) * (1.0 + frac)
