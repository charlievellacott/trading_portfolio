"""Reusable momentum series primitives and panel helpers."""

from __future__ import annotations

import pandas as pd

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
)

DEFAULT_LOOKBACK = 252
DEFAULT_SKIP = 21


def raw_momentum(
    close: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
) -> pd.Series:
    """
    Skip-month style momentum: ``close.shift(skip) / close.shift(lookback) - 1``.

    ``lookback`` (L) is how far back the start price is; ``skip`` (S) is how far
    back the end price is. Requires ``lookback > skip >= 0``.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if skip < 0:
        raise ValueError("skip must be >= 0")
    if lookback <= skip:
        raise ValueError("lookback must be greater than skip")
    return close.shift(skip) / close.shift(lookback) - 1.0


def add_raw_momentum(
    panel: pd.DataFrame,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    col: str = "raw_momentum",
) -> pd.DataFrame:
    """Return a copy of ``panel`` with per-ticker raw momentum in ``col``."""
    _require_columns(panel, {"date", "ticker", "close"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = work.groupby("ticker", sort=False)["close"].transform(
        lambda s: raw_momentum(s, lookback=lookback, skip=skip)
    )
    return _restore_order(work, original_index)
