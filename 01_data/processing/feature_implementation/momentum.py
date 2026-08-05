"""H-006 / H-007 momentum-family primitives: raw mom, near-52w, MAX lottery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    daily_simple_return,
    log_return,
)

DEFAULT_LOOKBACK = 252
DEFAULT_SKIP = 21

DEFAULT_NEAR_52W_WINDOW = 252
VALID_NEAR_52W_MODES = frozenset({"ratio", "log_drawdown"})

DEFAULT_MAX_N = 5
DEFAULT_MAX_WINDOW = 21
VALID_MAX_LOTTERY_MODES = frozenset({"simple", "log"})


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


# ---------------------------------------------------------------------------
# H-006 · 52-week high proximity
# ---------------------------------------------------------------------------


def near_52w_high(
    close: pd.Series,
    high: pd.Series,
    window: int = DEFAULT_NEAR_52W_WINDOW,
) -> pd.Series:
    """
    Proximity to rolling peak: ``close_t / max(high_{t-W+1}, …, high_t)``.

    Today is included in the peak window. Non-positive / non-finite ``close``
    or peak (``Hmax``) yields NaN. No floor is applied.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    c = close.astype(float)
    h = high.astype(float)
    hmax = h.rolling(window=window, min_periods=window).max()
    ok = (
        c.notna()
        & hmax.notna()
        & np.isfinite(c)
        & np.isfinite(hmax)
        & (c > 0)
        & (hmax > 0)
    )
    return (c / hmax).where(ok)


def apply_near_52w_mode(raw: pd.Series, mode: str) -> pd.Series:
    """Map raw near-52w ratio through ``mode`` (before optional CS rank)."""
    if mode not in VALID_NEAR_52W_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALID_NEAR_52W_MODES)}, got {mode!r}"
        )
    r = raw.astype(float)
    if mode == "ratio":
        return r
    return np.log(r.where(r > 0))


def add_near_52w_raw(
    panel: pd.DataFrame,
    *,
    window: int = DEFAULT_NEAR_52W_WINDOW,
    col: str = "near_52w_raw",
) -> pd.DataFrame:
    """Return a copy of ``panel`` with per-ticker near-52w ratio in ``col``."""
    _require_columns(panel, {"date", "ticker", "close", "high"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    if window < 1:
        raise ValueError("window must be >= 1")

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    hmax = work.groupby("ticker", sort=False)["high"].transform(
        lambda s: s.rolling(window=window, min_periods=window).max()
    )
    c = work["close"].astype(float)
    ok = (
        c.notna()
        & hmax.notna()
        & np.isfinite(c)
        & np.isfinite(hmax)
        & (c > 0)
        & (hmax > 0)
    )
    work[col] = (c / hmax).where(ok)
    return _restore_order(work, original_index)


# ---------------------------------------------------------------------------
# H-007 · MAX (lottery demand)
# ---------------------------------------------------------------------------


def max_lottery(
    returns: pd.Series,
    *,
    n_extreme: int = DEFAULT_MAX_N,
    window: int = DEFAULT_MAX_WINDOW,
) -> pd.Series:
    """
    Mean of the ``n_extreme`` largest finite returns in a trailing ``window``.

    Window ends at ``t`` (``min_periods=window``). Fewer than ``n_extreme``
    finite values in-window → NaN. No floor is applied.
    """
    if n_extreme < 1:
        raise ValueError("n_extreme must be >= 1")
    if window < 1:
        raise ValueError("window must be >= 1")
    if window < n_extreme:
        raise ValueError("window must be >= n_extreme")

    r = returns.astype(float)

    def _top_n_mean(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size < n_extreme:
            return np.nan
        # partition so the n_extreme largest are at the end
        part = np.partition(finite, -n_extreme)
        return float(np.mean(part[-n_extreme:]))

    return r.rolling(window=window, min_periods=window).apply(
        _top_n_mean, raw=True
    )


def add_max_lottery_raw(
    panel: pd.DataFrame,
    *,
    n_extreme: int = DEFAULT_MAX_N,
    window: int = DEFAULT_MAX_WINDOW,
    mode: str = "simple",
    col: str = "max_lottery_raw",
) -> pd.DataFrame:
    """Return a copy of ``panel`` with per-ticker MAX lottery signal in ``col``."""
    if mode not in VALID_MAX_LOTTERY_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALID_MAX_LOTTERY_MODES)}, got {mode!r}"
        )
    _require_columns(panel, {"date", "ticker", "close"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    if n_extreme < 1:
        raise ValueError("n_extreme must be >= 1")
    if window < n_extreme:
        raise ValueError("window must be >= n_extreme")

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())

    def _per_ticker(close: pd.Series) -> pd.Series:
        rets = (
            daily_simple_return(close)
            if mode == "simple"
            else log_return(close)
        )
        return max_lottery(rets, n_extreme=n_extreme, window=window)

    work[col] = work.groupby("ticker", sort=False)["close"].transform(_per_ticker)
    return _restore_order(work, original_index)
