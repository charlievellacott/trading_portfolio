"""H-008 Gross Profitability: series primitives and panel helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
)


# ---------------------------------------------------------------------------
# Series primitives
# ---------------------------------------------------------------------------


def gross_profitability(gp_asset: pd.Series) -> pd.Series:
    """
    Pass-through of precomputed ``gp_asset = gross_profit_ttm / assets``.

    Non-finite values become NaN. Bad assets (``<= 0``) are expected to already
    be NaN from the SEC fetcher (no floor here).
    """
    x = gp_asset.astype(float)
    return x.where(np.isfinite(x))


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def add_gross_profitability(
    panel: pd.DataFrame,
    *,
    col: str = "gross_profitability",
) -> pd.DataFrame:
    """
    Return a copy with raw ``gp_asset`` written to ``col``.

    Requires ``date``, ``ticker``, ``gp_asset`` (from
    ``fetch_gross_profitability_daily`` merge). Value is **before**
    cross-sectional normalization.
    """
    _require_columns(panel, {"date", "ticker", "gp_asset"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = gross_profitability(work["gp_asset"])
    return _restore_order(work, original_index)
