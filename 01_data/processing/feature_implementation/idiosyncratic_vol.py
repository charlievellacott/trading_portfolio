"""H-003 idiosyncratic vol: residual volatility from rolling market-beta OLS."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.linear_regression import rolling_ols_stats
from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    log_return,
    normalize_windows,
    regression_column_name,
)

DEFAULT_BETA_WINDOW = 20

_REQUIRED_PANEL = frozenset({"date", "ticker", "close"})


def idiosyncratic_vol(
    y: pd.Series,
    x: pd.Series,
    window: int,
) -> pd.Series:
    """
    Sample std (``ddof=1``) of in-window OLS residuals ``y ~ x``.

    Uses the same rolling OLS as ``linear_regression.rolling_ols_stats`` with
    ``include_idio_vol=True``.
    """
    stats = rolling_ols_stats(y, x, window, include_idio_vol=True)
    return stats["idio_vol"]


def add_idiosyncratic_vol(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: int | list[int] | tuple[int, ...] = DEFAULT_BETA_WINDOW,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Attach rolling idiosyncratic-volatility columns to a long OHLCV panel.

    Idiosyncratic vol is the sample std of OLS residuals from regressing each
    stock's log returns on market log returns over the rolling window ending at
    ``t``. Column naming mirrors ``beta.add_rolling_beta``: bare ``idio_vol``
    for one window; ``idio_vol_{w}`` when multiple windows are passed.

    This is the **raw** intermediate for H-003; the factor signal is the
    cross-sectional rank of idio vol (added later via feature store).
    """
    window_list = normalize_windows(windows)
    multi_window = len(window_list) > 1

    _require_columns(panel, _REQUIRED_PANEL)
    if market_col not in market_returns.columns:
        raise ValueError(f"market_returns missing column: {market_col!r}")
    if "date" not in market_returns.columns:
        raise ValueError("market_returns missing column: 'date'")

    if panel.empty:
        out = panel.copy()
        for window in window_list:
            col = regression_column_name("idio_vol", window, multi_window=multi_window)
            out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work["log_ret"] = work.groupby("ticker", sort=False)["close"].transform(log_return)
    work = work.merge(
        market_returns[["date", market_col]],
        on="date",
        how="left",
    )

    for window in window_list:
        col = regression_column_name("idio_vol", window, multi_window=multi_window)
        work[col] = np.nan
        for _, grp in work.groupby("ticker", sort=False):
            stats = rolling_ols_stats(
                grp["log_ret"],
                grp[market_col],
                window,
                include_idio_vol=True,
            )
            work.loc[grp.index, col] = stats["idio_vol"].to_numpy()

    drop_cols = ["log_ret", market_col]
    result = work.drop(columns=[c for c in drop_cols if c in work.columns])
    return _restore_order(result, original_index)
