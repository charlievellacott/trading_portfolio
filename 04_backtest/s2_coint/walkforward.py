"""S2 expanding purged walk-forward on bar dates (not S1 week-starts)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class S2WalkForwardFold:
    fold_id: int
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    embargo_dates: pd.DatetimeIndex


def make_s2_folds(
    is_dates: pd.DatetimeIndex | pd.Series,
    *,
    n_folds: int = 3,
    embargo_bars: int = 5,
) -> list[S2WalkForwardFold]:
    """Expanding chronological folds on research-IS bar timestamps.

    Split unique sorted dates into ``n_folds + 1`` blocks. Fold ``k`` validates
    on block ``k + 1`` and trains on all earlier dates, dropping ``embargo_bars``
    immediately before the val window.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if embargo_bars < 0:
        raise ValueError(f"embargo_bars must be >= 0, got {embargo_bars}")

    dates = pd.DatetimeIndex(pd.to_datetime(is_dates)).sort_values().unique()
    n = len(dates)
    n_blocks = n_folds + 1
    min_needed = n_blocks * max(embargo_bars + 5, 8)
    if n < min_needed:
        raise ValueError(
            f"IS bars={n} too short for n_folds={n_folds} "
            f"+ embargo_bars={embargo_bars} (need ~{min_needed})"
        )

    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    blocks = [dates[edges[i] : edges[i + 1]] for i in range(n_blocks)]
    for i, blk in enumerate(blocks):
        if len(blk) == 0:
            raise ValueError(f"empty date block {i} in walk-forward split")

    folds: list[S2WalkForwardFold] = []
    for k in range(n_folds):
        val_dates = pd.DatetimeIndex(blocks[k + 1])
        train_pool = dates[dates < val_dates.min()]
        if len(train_pool) <= embargo_bars:
            raise ValueError(
                f"fold {k + 1}: train pool length {len(train_pool)} "
                f"<= embargo_bars={embargo_bars}"
            )
        if embargo_bars > 0:
            embargo_dates = pd.DatetimeIndex(train_pool[-embargo_bars:])
            train_dates = pd.DatetimeIndex(train_pool[:-embargo_bars])
        else:
            embargo_dates = pd.DatetimeIndex([])
            train_dates = pd.DatetimeIndex(train_pool)
        if len(train_dates) == 0 or len(val_dates) == 0:
            raise ValueError(f"fold {k + 1}: empty train or val dates")
        folds.append(
            S2WalkForwardFold(
                fold_id=k + 1,
                train_dates=train_dates,
                val_dates=val_dates,
                embargo_dates=embargo_dates,
            )
        )
    return folds


def embargo_bars_for_config(*, bar: str) -> int:
    """5 trading days on 1D; one calendar week of 1H bars (≈5 sessions × 6 hours)."""
    if bar == "1h":
        return 5 * 6
    return 5
