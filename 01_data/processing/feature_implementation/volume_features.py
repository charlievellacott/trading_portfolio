"""Abnormal volume (trading attention): series primitives and panel helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    abnormal_z,
    normalize_windows,
    windowed_column_name,
)


# ---------------------------------------------------------------------------
# Series primitives
# ---------------------------------------------------------------------------


def abnormal_volume_z(
    volume: pd.Series,
    *,
    smooth_window: int = 5,
    baseline_window: int = 60,
) -> pd.Series:
    """Own-history z of ``log1p(volume)`` (same pattern as H-009/H-010)."""
    x = np.log1p(volume.astype(float).clip(lower=0))
    return abnormal_z(
        x, smooth_window=smooth_window, baseline_window=baseline_window
    )


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def add_abnormal_volume(
    panel: pd.DataFrame,
    *,
    smooth_window: int = 5,
    baseline_window: int = 60,
    col: str = "abnormal_volume",
) -> pd.DataFrame:
    """Attach abnormal volume column (raw own-history z; no CS normalize)."""
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")
    if baseline_window < 1:
        raise ValueError("baseline_window must be >= 1")
    _require_columns(panel, {"date", "ticker", "volume"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    parts = []
    for _, grp in work.groupby("ticker", sort=False):
        parts.append(
            abnormal_volume_z(
                grp["volume"],
                smooth_window=smooth_window,
                baseline_window=baseline_window,
            )
        )
    work[col] = pd.concat(parts).reindex(work.index)
    return _restore_order(work, original_index)


def add_abnormal_volume_multi(
    panel: pd.DataFrame,
    *,
    smooth_window: int | list[int] | tuple[int, ...] = 5,
    baseline_window: int | list[int] | tuple[int, ...] = 60,
) -> pd.DataFrame:
    """
    Add ``abnormal_volume`` [``_{S}_{B}``] for each smooth/baseline combo.

    One combo → bare ``abnormal_volume``; multiple → suffixed columns.
    """
    smooth_list = normalize_windows(smooth_window)
    baseline_list = normalize_windows(baseline_window)
    combos = [(s, b) for s in smooth_list for b in baseline_list]
    multi = len(combos) > 1
    result = panel
    for s, b in combos:
        col = windowed_column_name("abnormal_volume", s, b, multi=multi)
        result = add_abnormal_volume(
            result, smooth_window=s, baseline_window=b, col=col,
        )
    return result
