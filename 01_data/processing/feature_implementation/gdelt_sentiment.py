"""H-009 GDELT sentiment: series primitives and panel helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    abnormal_z,
)


# ---------------------------------------------------------------------------
# Series primitives
# ---------------------------------------------------------------------------


def rolling_median_tone(tone: pd.Series, *, window: int = 5) -> pd.Series:
    """Rolling median of daily median tone."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return tone.astype(float).rolling(window, min_periods=window).median()


def rolling_log_attention(n_articles: pd.Series, *, window: int = 5) -> pd.Series:
    """Rolling mean of ``log1p(n_articles)``."""
    if window < 1:
        raise ValueError("window must be >= 1")
    x = np.log1p(n_articles.astype(float).clip(lower=0))
    return x.rolling(window, min_periods=window).mean()


def tone_attention_interact(
    tone_roll: pd.Series,
    attention_roll: pd.Series,
) -> pd.Series:
    """``tone_roll * attention_roll`` (attention already log1p-smoothed)."""
    return tone_roll.astype(float) * attention_roll.astype(float)


def tone_momentum(
    tone: pd.Series,
    *,
    short_window: int = 5,
    long_window: int = 21,
) -> pd.Series:
    """Short rolling median tone minus long rolling median tone."""
    if short_window < 1 or long_window < 1:
        raise ValueError("short_window and long_window must be >= 1")
    if long_window <= short_window:
        raise ValueError("long_window must be > short_window")
    short = rolling_median_tone(tone, window=short_window)
    long = rolling_median_tone(tone, window=long_window)
    return short - long


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _per_ticker(panel: pd.DataFrame, col: str, fn) -> pd.Series:
    parts = []
    for _, grp in panel.groupby("ticker", sort=False):
        parts.append(fn(grp[col]))
    return pd.concat(parts).reindex(panel.index)


def add_gdelt_tone(
    panel: pd.DataFrame,
    *,
    window: int = 5,
    tone_col: str = "median_tone",
    col: str = "gdelt_tone",
) -> pd.DataFrame:
    """Attach rolling median tone column."""
    _require_columns(panel, {"date", "ticker", tone_col})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    ordered = _sorted_by_ticker_date(panel)
    orig = panel.index
    result = ordered.copy()
    result[col] = _per_ticker(
        result, tone_col, lambda s: rolling_median_tone(s, window=window)
    )
    return _restore_order(result, orig)


def add_gdelt_attention(
    panel: pd.DataFrame,
    *,
    window: int = 5,
    n_col: str = "n_articles",
    col: str = "gdelt_attention",
) -> pd.DataFrame:
    """Attach rolling log1p attention column."""
    _require_columns(panel, {"date", "ticker", n_col})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    ordered = _sorted_by_ticker_date(panel)
    orig = panel.index
    result = ordered.copy()
    result[col] = _per_ticker(
        result, n_col, lambda s: rolling_log_attention(s, window=window)
    )
    return _restore_order(result, orig)


def add_gdelt_abnormal_tone(
    panel: pd.DataFrame,
    *,
    smooth_window: int = 5,
    baseline_window: int = 60,
    tone_col: str = "median_tone",
    col: str = "gdelt_abnormal_tone",
) -> pd.DataFrame:
    """Attach abnormal (own-history z) tone column."""
    _require_columns(panel, {"date", "ticker", tone_col})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    ordered = _sorted_by_ticker_date(panel)
    orig = panel.index
    result = ordered.copy()
    result[col] = _per_ticker(
        result,
        tone_col,
        lambda s: abnormal_z(
            s, smooth_window=smooth_window, baseline_window=baseline_window
        ),
    )
    return _restore_order(result, orig)


def add_gdelt_abnormal_attention(
    panel: pd.DataFrame,
    *,
    smooth_window: int = 5,
    baseline_window: int = 60,
    n_col: str = "n_articles",
    col: str = "gdelt_abnormal_attention",
) -> pd.DataFrame:
    """Attach abnormal (own-history z) of daily ``log1p(n_articles)``."""
    _require_columns(panel, {"date", "ticker", n_col})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    ordered = _sorted_by_ticker_date(panel)
    orig = panel.index
    result = ordered.copy()

    def _fn(s: pd.Series) -> pd.Series:
        x = np.log1p(s.astype(float).clip(lower=0))
        return abnormal_z(
            x, smooth_window=smooth_window, baseline_window=baseline_window
        )

    result[col] = _per_ticker(result, n_col, _fn)
    return _restore_order(result, orig)


def add_gdelt_tone_x_attention(
    panel: pd.DataFrame,
    *,
    window: int = 5,
    tone_col: str = "median_tone",
    n_col: str = "n_articles",
    col: str = "gdelt_tone_x_attention",
) -> pd.DataFrame:
    """Attach tone × attention interaction on rolling windows."""
    _require_columns(panel, {"date", "ticker", tone_col, n_col})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    ordered = _sorted_by_ticker_date(panel)
    orig = panel.index
    result = ordered.copy()
    parts = []
    for _, grp in result.groupby("ticker", sort=False):
        t = rolling_median_tone(grp[tone_col], window=window)
        a = rolling_log_attention(grp[n_col], window=window)
        parts.append(tone_attention_interact(t, a))
    result[col] = pd.concat(parts).reindex(result.index)
    return _restore_order(result, orig)


def add_gdelt_tone_mom(
    panel: pd.DataFrame,
    *,
    short_window: int = 5,
    long_window: int = 21,
    tone_col: str = "median_tone",
    col: str = "gdelt_tone_mom",
) -> pd.DataFrame:
    """Attach tone momentum (short − long rolling median)."""
    _require_columns(panel, {"date", "ticker", tone_col})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    ordered = _sorted_by_ticker_date(panel)
    orig = panel.index
    result = ordered.copy()
    result[col] = _per_ticker(
        result,
        tone_col,
        lambda s: tone_momentum(
            s, short_window=short_window, long_window=long_window
        ),
    )
    return _restore_order(result, orig)
