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


def windowed_column_name(stem: str, *parts: int, multi: bool) -> str:
    """Bare ``stem`` when one combo; ``stem_{p0}_{p1}_...`` when ``multi``."""
    if not multi:
        return stem
    if not parts:
        raise ValueError("parts must be non-empty when multi=True")
    return stem + "".join(f"_{p}" for p in parts)


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


# ---------------------------------------------------------------------------
# H-004 primitives (added below existing code — no alterations above)
# ---------------------------------------------------------------------------


def rolling_conditional_ols_stats(
    y: pd.Series,
    x: pd.Series,
    window: int,
    *,
    side: str,
    min_obs: int | None = None,
) -> pd.DataFrame:
    """
    Rolling OLS of ``y`` on ``x`` restricted to bars where ``x`` is below
    (``side='down'``) or at/above (``side='up'``) the in-window mean of ``x``.

    Parameters
    ----------
    side : {'down', 'up'}
    min_obs : int or None
        Minimum observations in the conditional subset for the regression to be
        valid. Default ``max(20, window // 4)``.

    Returns DataFrame with columns ``alpha``, ``beta``, ``r2``, ``n_obs``.
    """
    if side not in ("down", "up"):
        raise ValueError(f"side must be 'down' or 'up', got {side!r}")
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_obs is None:
        min_obs = max(20, window // 4)
    if min_obs < 2:
        min_obs = 2

    n = len(y)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    n_obs_arr = np.full(n, np.nan)

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

        x_mean = x_v.mean()
        if side == "down":
            mask = x_v < x_mean
        else:
            mask = x_v >= x_mean

        count = int(mask.sum())
        if count < min_obs:
            continue

        y_sub = y_v[mask]
        x_sub = x_v[mask]
        x_demean = x_sub - x_sub.mean()
        ss_xx = np.sum(x_demean ** 2)
        if ss_xx == 0.0:
            continue

        y_demean = y_sub - y_sub.mean()
        b = np.sum(x_demean * y_demean) / ss_xx
        a = y_sub.mean() - b * x_sub.mean()
        eps = y_sub - a - b * x_sub
        ss_res = np.sum(eps ** 2)
        ss_tot = np.sum(y_demean ** 2)

        alpha[i] = a
        beta[i] = b
        r2[i] = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        n_obs_arr[i] = count

    return pd.DataFrame(
        {"alpha": alpha, "beta": beta, "r2": r2, "n_obs": n_obs_arr},
        index=y.index,
    )


def rolling_multi_ols_stats(
    y: pd.Series,
    X: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Rolling multivariate OLS of ``y`` on columns of ``X`` with intercept.

    Uses the last ``window`` jointly finite rows. NaN when the design matrix
    is rank-deficient or fewer than ``window`` valid observations exist.

    Returns DataFrame with columns ``["alpha", *X.columns]`` (intercept + slopes).
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    regressors = list(X.columns)
    k = len(regressors)
    n = len(y)

    out_alpha = np.full(n, np.nan)
    out_slopes = np.full((n, k), np.nan)

    y_arr = y.to_numpy(dtype=float)
    X_arr = X.to_numpy(dtype=float)

    for i in range(window - 1, n):
        y_w = y_arr[i - window + 1 : i + 1]
        X_w = X_arr[i - window + 1 : i + 1]
        valid = np.isfinite(y_w) & np.all(np.isfinite(X_w), axis=1)
        if int(valid.sum()) < window:
            continue

        y_v = y_w[valid]
        X_v = X_w[valid]
        ones = np.ones((len(y_v), 1))
        design = np.hstack([ones, X_v])

        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue

        # Normal equations: (X'X)^-1 X'y
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue

        out_alpha[i] = coeffs[0]
        out_slopes[i] = coeffs[1:]

    result: dict[str, np.ndarray] = {"alpha": out_alpha}
    for j, col in enumerate(regressors):
        result[col] = out_slopes[:, j]
    return pd.DataFrame(result, index=y.index)


def blume_adjust(
    beta: pd.Series,
    *,
    alpha: float = 0.67,
    beta_prior: float = 1.0,
) -> pd.Series:
    """Blume-adjusted beta: ``alpha * beta + (1 - alpha) * beta_prior``."""
    return alpha * beta.astype(float) + (1.0 - alpha) * beta_prior


def residual_momentum_signal(
    residuals: pd.Series,
    formation_window: int,
    skip: int,
) -> pd.Series:
    """
    Blitz residual-momentum signal: ``mean(ε) / std(ε)`` over the formation
    window excluding the most recent ``skip`` bars.

    For each bar ``t``, uses residuals from ``t - formation_window + 1`` to
    ``t - skip`` (inclusive). Returns NaN when std == 0 or insufficient data.

    Requires ``formation_window > skip >= 0``.
    """
    if formation_window < 1:
        raise ValueError("formation_window must be >= 1")
    if skip < 0:
        raise ValueError("skip must be >= 0")
    if formation_window <= skip:
        raise ValueError("formation_window must be greater than skip")

    usable_length = formation_window - skip
    res = residuals.astype(float)
    n = len(res)
    out = np.full(n, np.nan)
    arr = res.to_numpy()

    for i in range(n):
        start = i - formation_window + 1
        end = i - skip + 1
        if start < 0:
            continue
        window_vals = arr[start:end]
        finite = window_vals[np.isfinite(window_vals)]
        if len(finite) < usable_length:
            continue
        std = float(np.std(finite, ddof=1))
        if std == 0.0:
            continue
        out[i] = float(np.mean(finite)) / std

    return pd.Series(out, index=residuals.index, dtype=float)
