"""Feature-store entrypoints that add alpha columns to long OHLCV panels."""

from __future__ import annotations

import pandas as pd

from data.processing.feature_implementation.gk_vol_ratio import (
    VALID_MODES as GK_VALID_MODES,
    add_gk_realised_ratio_raw,
    apply_ratio_mode,
)
from data.processing.feature_implementation.obv_momentum import (
    VALID_MODES,
    add_obv_confirmed_combined,
    cross_sectional_pct_rank,
)


def add_obv_confirmed_momentum(
    panel: pd.DataFrame,
    *,
    lookback: int = 252,
    skip: int = 21,
    obv_window: int = 20,
    mode: str = "signed",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add H-001 OBV-confirmed momentum as column ``obv_mom_{mode}``.

    Features at date ``t`` use OHLCV through the close of ``t``. The intended
    prediction target is the next close (``P_{t+1} / P_t - 1``); labels are not
    added here.

    Parameters
    ----------
    panel:
        Long-format frame with ``date``, ``ticker``, ``close``, ``volume``.
    lookback, skip:
        Momentum windows (L, S): ``P_{t-S} / P_{t-L} - 1``.
    obv_window:
        OBV trend window W: ``OBV_t - OBV_{t-W}``.
    mode:
        ``"signed"`` (default), ``"strict_zero"``, or ``"soft"``.
    normalize:
        If True, store cross-sectional percentile rank of the combined signal
        within each date (GBM-ready). If False, store the raw combined signal.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if skip < 0:
        raise ValueError("skip must be >= 0")
    if lookback <= skip:
        raise ValueError("lookback must be greater than skip")
    if obv_window < 1:
        raise ValueError("obv_window must be >= 1")

    required = {"date", "ticker", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    out_col = f"obv_mom_{mode}"
    if panel.empty:
        out = panel.copy()
        out[out_col] = pd.Series(dtype=float)
        return out

    combined_col = "_obv_confirmed_combined_tmp"
    result = add_obv_confirmed_combined(
        panel,
        lookback=lookback,
        skip=skip,
        obv_window=obv_window,
        mode=mode,
        col=combined_col,
    )

    if normalize:
        result[out_col] = cross_sectional_pct_rank(result, combined_col)
    else:
        result[out_col] = result[combined_col]

    return result.drop(columns=[combined_col])


def add_gk_vol_ratio(
    panel: pd.DataFrame,
    *,
    gk_window: int = 5,
    realised_window: int = 20,
    mode: str = "ratio",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add H-002 GK / realised vol ratio as column ``gk_vol_{mode}``.

    Features at date ``t`` use OHLC through the close of ``t``. The intended
    prediction target is the next close (``P_{t+1} / P_t - 1``); labels are not
    added here. No denominator floor and no winsorization are applied.

    Parameters
    ----------
    panel:
        Long-format frame with ``date``, ``ticker``, ``open``, ``high``,
        ``low``, ``close``.
    gk_window:
        Short window for the mean of daily Garman–Klass volatility.
    realised_window:
        Number of log close-to-close returns in the realised-vol std
        (ending at ``t``).
    mode:
        ``"ratio"`` (default), ``"log_ratio"``, or ``"reversal"``.
    normalize:
        If True, store cross-sectional percentile rank of the mode-transformed
        signal within each date (GBM-ready). If False, store the unranked value.
    """
    if mode not in GK_VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(GK_VALID_MODES)}, got {mode!r}")
    if gk_window < 1:
        raise ValueError("gk_window must be >= 1")
    if realised_window < 1:
        raise ValueError("realised_window must be >= 1")

    required = {"date", "ticker", "open", "high", "low", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    out_col = f"gk_vol_{mode}"
    if panel.empty:
        out = panel.copy()
        out[out_col] = pd.Series(dtype=float)
        return out

    raw_col = "_gk_realised_ratio_raw_tmp"
    mode_col = "_gk_vol_mode_tmp"
    result = add_gk_realised_ratio_raw(
        panel,
        gk_window=gk_window,
        realised_window=realised_window,
        col=raw_col,
    )
    result[mode_col] = apply_ratio_mode(result[raw_col], mode=mode)

    if normalize:
        result[out_col] = cross_sectional_pct_rank(result, mode_col)
    else:
        result[out_col] = result[mode_col]

    return result.drop(columns=[raw_col, mode_col])
