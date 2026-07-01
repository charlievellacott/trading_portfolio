"""Clean long-format OHLCV panels from equity fetchers."""

from __future__ import annotations

import pandas as pd


def cap_cross_sectional_outliers(
    panel: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    min_obs: int = 3,
) -> pd.DataFrame:
    """
    Winsorize extreme values using only the cross-section available on each date.

    Only ``columns`` are modified; defaults to ``["close"]``.
    """
    cols = ["close"] if columns is None else columns
    if lower_q < 0 or upper_q > 1 or lower_q >= upper_q:
        raise ValueError("require 0 <= lower_q < upper_q <= 1")
    if min_obs < 1:
        raise ValueError("min_obs must be >= 1")

    required = {"date", "ticker", *cols}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    if panel.empty:
        return panel.copy()

    result = panel.copy()
    for col in cols:
        for _, group in result.groupby("date", sort=False):
            values = group[col]
            if values.notna().sum() < min_obs:
                continue
            lower, upper = values.quantile([lower_q, upper_q])
            result.loc[group.index, col] = values.clip(lower=lower, upper=upper)

    return result


def forward_fill_panel(
    panel: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Forward-fill missing values within each ticker on existing rows only.

    Only ``columns`` are modified; defaults to ``["close"]``.
    No business-day reindex and no backfill.
    """
    cols = ["close"] if columns is None else columns
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided")

    required = {"date", "ticker", *cols}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    if panel.empty:
        return panel.copy()

    result = panel.sort_values(["ticker", "date"], kind="mergesort").copy()
    result.loc[:, cols] = result.groupby("ticker", sort=False)[cols].ffill(limit=limit)
    return result.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
