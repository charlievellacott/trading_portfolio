"""Rolling market-beta OLS primitives and panel helpers for walk-forward use."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BETA_WINDOW = 20

_REQUIRED_PANEL = frozenset({"date", "ticker", "close"})
_OLS_OUTPUT_COLUMNS = ("alpha", "beta", "r2", "idio_vol")


def log_return(close: pd.Series) -> pd.Series:
    """``ln(C_t / C_{t-1})``; first bar is NaN."""
    c = close.astype(float)
    return np.log(c / c.shift(1))


def normalize_windows(windows: int | list[int] | tuple[int, ...]) -> list[int]:
    """Normalize ``windows`` to a non-empty list of positive ints."""
    if isinstance(windows, bool):
        raise ValueError("windows must be a positive int or a list of positive ints")
    if isinstance(windows, int):
        items = [windows]
    elif isinstance(windows, (list, tuple)):
        items = list(windows)
    else:
        raise ValueError("windows must be a positive int or a list of positive ints")
    if not items:
        raise ValueError("windows must be a non-empty list of positive ints")
    for w in items:
        if not isinstance(w, int) or isinstance(w, bool) or w < 1:
            raise ValueError(f"windows entries must be positive ints, got {w!r}")
    return items


def regression_column_name(metric: str, window: int, *, multi_window: bool) -> str:
    """Bare ``metric`` when one window; ``metric_{window}`` when multiple."""
    if multi_window:
        return f"{metric}_{window}"
    return metric


def rolling_ols_stats(
    y: pd.Series,
    x: pd.Series,
    window: int,
    *,
    include_idio_vol: bool = False,
) -> pd.DataFrame:
    """
    Rolling OLS of ``y`` on ``x`` ending at each bar ``t``.

    Uses the last ``window`` jointly finite ``(y, x)`` pairs with
    ``min_periods=window``. ``Var(x)==0`` or ``SS_tot==0`` → NaN outputs.
  """
    if window < 1:
        raise ValueError("window must be >= 1")

    n = len(y)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    idio_vol = np.full(n, np.nan) if include_idio_vol else None

    y_arr = y.to_numpy(dtype=float)
    x_arr = x.to_numpy(dtype=float)

    for i in range(window - 1, n):
        y_w = y_arr[i - window + 1 : i + 1]
        x_w = x_arr[i - window + 1 : i + 1]
        valid = np.isfinite(y_w) & np.isfinite(x_w)
        if int(valid.sum()) < window:
            continue
        y_v = y_w[valid]
        x_v = x_w[valid]
        x_demean = x_v - x_v.mean()
        ss_xx = np.sum(x_demean ** 2)
        if ss_xx == 0.0:
            continue
        y_demean = y_v - y_v.mean()
        b = np.sum(x_demean * y_demean) / ss_xx
        a = y_v.mean() - b * x_v.mean()
        eps = y_v - a - b * x_v
        ss_res = np.sum(eps ** 2)
        ss_tot = np.sum(y_demean ** 2)
        alpha[i] = a
        beta[i] = b
        r2[i] = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        if include_idio_vol and idio_vol is not None:
            idio_vol[i] = float(np.std(eps, ddof=1))

    out: dict[str, np.ndarray] = {"alpha": alpha, "beta": beta, "r2": r2}
    if include_idio_vol and idio_vol is not None:
        out["idio_vol"] = idio_vol
    return pd.DataFrame(out, index=y.index)


def rolling_residual(
    y: pd.Series,
    x: pd.Series,
    alpha: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """Per-bar residual ``y - alpha - beta * x`` using aligned rolling coefficients."""
    return y.astype(float) - alpha.astype(float) - beta.astype(float) * x.astype(float)


def market_return_frame(
    market_panel: pd.DataFrame,
    *,
    close_col: str = "close",
    out_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Build ``date`` + market log-return column from a single-ticker OHLCV panel.

    Parameters
    ----------
    market_panel:
        Long-format frame with ``date`` and ``close`` (e.g. SPY from ``fetch_ohlcv``).
    """
    required = {"date", close_col}
    missing = required - set(market_panel.columns)
    if missing:
        raise ValueError(f"market_panel missing columns: {sorted(missing)}")
    if market_panel.empty:
        return pd.DataFrame(columns=["date", out_col])

    out = (
        market_panel.sort_values("date")
        .assign(**{out_col: lambda d: log_return(d[close_col])})
        [["date", out_col]]
        .reset_index(drop=True)
    )
    return out


def _require_columns(panel: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")


def _sorted_by_ticker_date(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(["ticker", "date"], kind="mergesort")


def _restore_order(result: pd.DataFrame, original_index: pd.Index) -> pd.DataFrame:
    return result.reindex(original_index)


def add_rolling_beta(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: int | list[int] | tuple[int, ...] = DEFAULT_BETA_WINDOW,
    market_col: str = "market_log_ret",
    include_r2: bool = True,
) -> pd.DataFrame:
    """
    Attach rolling OLS alpha / beta / r2 to a long OHLCV panel.

    Features at date ``t`` use stock and market log returns through the close of
    ``t`` only. Column names are bare (``alpha``, ``beta``, ``r2``) when
    ``len(windows)==1``; suffixed (``alpha_{w}``, …) when multiple windows.

    Parameters
    ----------
    panel:
        Long-format frame with ``date``, ``ticker``, ``close``.
    market_returns:
        Frame with ``date`` and ``market_col`` (from ``market_return_frame``).
    windows:
        Single int or list of rolling window lengths.
    market_col:
        Column name in ``market_returns`` for benchmark log returns.
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
        metrics = ["alpha", "beta"]
        if include_r2:
            metrics.append("r2")
        for window in window_list:
            for metric in metrics:
                out[regression_column_name(metric, window, multi_window=multi_window)] = (
                    pd.Series(dtype=float)
                )
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
        alpha_col = regression_column_name("alpha", window, multi_window=multi_window)
        beta_col = regression_column_name("beta", window, multi_window=multi_window)
        r2_col = regression_column_name("r2", window, multi_window=multi_window)

        work[alpha_col] = np.nan
        work[beta_col] = np.nan
        if include_r2:
            work[r2_col] = np.nan

        for _, grp in work.groupby("ticker", sort=False):
            stats = rolling_ols_stats(grp["log_ret"], grp[market_col], window)
            work.loc[grp.index, alpha_col] = stats["alpha"].to_numpy()
            work.loc[grp.index, beta_col] = stats["beta"].to_numpy()
            if include_r2:
                work.loc[grp.index, r2_col] = stats["r2"].to_numpy()

    drop_cols = ["log_ret", market_col]
    result = work.drop(columns=[c for c in drop_cols if c in work.columns])
    return _restore_order(result, original_index)
