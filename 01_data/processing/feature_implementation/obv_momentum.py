"""H-001 OBV-confirmed momentum: series primitives and panel helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.momentum import raw_momentum
from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    cross_sectional_pct_rank,
)

VALID_MODES = frozenset({"signed", "strict_zero", "soft"})
DEFAULT_LOOKBACK = 252
DEFAULT_SKIP = 21
DEFAULT_OBV_WINDOW = 20

_REQUIRED_OHLCV = frozenset({"date", "ticker", "close", "volume"})


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Cumulative signed volume; flat close days contribute 0.

    ``OBV_t = OBV_{t-1} + sign(close_t - close_{t-1}) * volume_t``
    """
    delta = close.diff()
    signed = np.sign(delta.to_numpy(dtype=float))
    signed = np.where(np.isnan(signed), np.nan, signed)
    contribution = pd.Series(signed, index=close.index, dtype=float) * volume.astype(float)
    contribution = contribution.fillna(0.0)
    return contribution.cumsum()


def obv_trend(obv: pd.Series, window: int = DEFAULT_OBV_WINDOW) -> pd.Series:
    """``OBV_t - OBV_{t-window}``."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return obv - obv.shift(window)


def signs_agree(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    True when ``sign(a) == sign(b)`` and both are nonzero.

    Zero vs nonzero (or either NaN) counts as disagreement / not agreeing.
    """
    sa = np.sign(a.to_numpy(dtype=float))
    sb = np.sign(b.to_numpy(dtype=float))
    agree = (sa == sb) & (sa != 0) & (sb != 0) & np.isfinite(sa) & np.isfinite(sb)
    return pd.Series(agree, index=a.index, dtype=bool)


def combine_momentum_obv(
    mom: pd.Series,
    obv_tr: pd.Series,
    mode: str,
    soft_weight: pd.Series | None = None,
) -> pd.Series:
    """
    Combine raw momentum with OBV trend under ``mode``.

    - ``signed``: keep momentum when signs agree, else flip sign (keep magnitude).
    - ``strict_zero``: keep momentum when signs agree, else 0.
    - ``soft``: ``mom * soft_weight``; ``soft_weight`` must be provided
      (typically ``2 * cs_pctrank(obv_trend) - 1``).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    if mode == "soft":
        if soft_weight is None:
            raise ValueError("soft_weight is required when mode='soft'")
        out = mom.astype(float) * soft_weight.astype(float)
        return out.where(mom.notna() & obv_tr.notna() & soft_weight.notna())

    agree = signs_agree(mom, obv_tr)
    mom_f = mom.astype(float)
    if mode == "signed":
        combined = pd.Series(
            np.where(agree.to_numpy(), mom_f.to_numpy(), -mom_f.to_numpy()),
            index=mom.index,
            dtype=float,
        )
    else:
        # strict_zero
        combined = pd.Series(
            np.where(agree.to_numpy(), mom_f.to_numpy(), 0.0),
            index=mom.index,
            dtype=float,
        )
    return combined.where(mom.notna() & obv_tr.notna())


def _per_ticker_obv(close: pd.Series, volume: pd.Series, ticker: pd.Series) -> pd.Series:
    """Compute OBV within each ticker, preserving row alignment."""
    out = pd.Series(np.nan, index=close.index, dtype=float)
    for _, idx in ticker.groupby(ticker, sort=False).groups.items():
        out.loc[idx] = on_balance_volume(close.loc[idx], volume.loc[idx]).to_numpy()
    return out


def add_obv(
    panel: pd.DataFrame,
    *,
    col: str = "obv",
) -> pd.DataFrame:
    """Return a copy of ``panel`` with per-ticker on-balance volume in ``col``."""
    _require_columns(panel, {"date", "ticker", "close", "volume"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work[col] = _per_ticker_obv(work["close"], work["volume"], work["ticker"])
    return _restore_order(work, original_index)


def add_obv_trend(
    panel: pd.DataFrame,
    *,
    obv_window: int = DEFAULT_OBV_WINDOW,
    obv_col: str = "obv",
    col: str = "obv_trend",
) -> pd.DataFrame:
    """
    Return a copy of ``panel`` with OBV trend in ``col``.

    Computes OBV into ``obv_col`` first when that column is missing.
    """
    _require_columns(panel, {"date", "ticker", "close", "volume"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    work = panel if obv_col in panel.columns else add_obv(panel, col=obv_col)
    original_index = work.index
    work = _sorted_by_ticker_date(work.copy())
    work[col] = work.groupby("ticker", sort=False)[obv_col].transform(
        lambda s: obv_trend(s, window=obv_window)
    )
    return _restore_order(work, original_index)


def add_obv_confirmed_combined(
    panel: pd.DataFrame,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    obv_window: int = DEFAULT_OBV_WINDOW,
    mode: str = "signed",
    col: str = "obv_confirmed_combined",
) -> pd.DataFrame:
    """
    Return a copy of ``panel`` with the combined OBV-confirmed signal in ``col``.

    Combined value is **before** cross-sectional normalization.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    _require_columns(panel, _REQUIRED_OHLCV)

    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())

    mom = work.groupby("ticker", sort=False)["close"].transform(
        lambda s: raw_momentum(s, lookback=lookback, skip=skip)
    )
    obv = _per_ticker_obv(work["close"], work["volume"], work["ticker"])
    obv_tr = obv.groupby(work["ticker"], sort=False).transform(
        lambda s: obv_trend(s, window=obv_window)
    )

    soft_weight: pd.Series | None = None
    if mode == "soft":
        tmp = work[["date"]].copy()
        tmp["_obv_tr"] = obv_tr.to_numpy()
        soft_weight = 2.0 * cross_sectional_pct_rank(tmp, "_obv_tr") - 1.0

    work[col] = combine_momentum_obv(mom, obv_tr, mode=mode, soft_weight=soft_weight)
    return _restore_order(work, original_index)
