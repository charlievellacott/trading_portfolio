"""H-004 beta features: workspace pattern, panel helpers, and column parser."""

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

from data.processing.feature_implementation.linear_regression import (
    rolling_conditional_ols_stats,
    rolling_multi_ols_stats,
    rolling_ols_stats,
    rolling_residual,
)
from data.processing.feature_implementation.utilities import (
    _require_columns,
    _restore_order,
    _sorted_by_ticker_date,
    cross_sectional_pct_rank,
    log_return,
    normalize_windows,
    regression_column_name,
    windowed_column_name,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BETA_WINDOW = 20
_OLS_OUTPUT_COLUMNS = ("alpha", "beta", "r2", "idio_vol")

_WS_PREFIX = "_ws_"

_REQUIRED_PANEL = frozenset({"date", "ticker", "close"})
_FF_REQUIRED_COLS = frozenset({"date", "mkt_rf", "smb", "hml", "mom", "rf"})

_SINGLE_WINDOW_FEATURES = frozenset({
    "beta", "downside_beta", "upside_beta", "net_beta_spread",
    "rel_downside_beta", "rel_upside_beta", "blume_beta",
    "smart_beta_smb", "smart_beta_hml", "smart_beta_mom",
})
_DUAL_WINDOW_FEATURES = frozenset({
    "residual_mom", "smart_residual_mom",
})
ALL_H004_FEATURES = _SINGLE_WINDOW_FEATURES | _DUAL_WINDOW_FEATURES

_ALL_STEMS_SORTED = sorted(ALL_H004_FEATURES, key=lambda s: -len(s))


# ---------------------------------------------------------------------------
# Beta-specific primitives (moved from beta.py)
# ---------------------------------------------------------------------------


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
    Blitz residual-momentum signal: ``mean(e) / std(e)`` over the formation
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
    ``len(windows)==1``; suffixed (``alpha_{w}``, ...) when multiple windows.

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
# Workspace column naming
# ---------------------------------------------------------------------------

def _ws_col(metric: str, window: int) -> str:
    """Workspace column: always suffixed (e.g. '_ws_beta_252')."""
    return f"{_WS_PREFIX}{metric}_{window}"


def _spy_ws_expected(windows: list[int]) -> list[str]:
    """All SPY workspace column names for the given windows."""
    metrics = ("beta", "alpha", "residual", "downside_beta", "upside_beta")
    return [_ws_col(m, w) for w in windows for m in metrics]


def _ff_ws_expected(windows: list[int]) -> list[str]:
    """All FF workspace column names for the given windows."""
    metrics = ("smart_beta_smb", "smart_beta_hml", "smart_beta_mom", "ff4_residual")
    return [_ws_col(m, w) for w in windows for m in metrics]


# ---------------------------------------------------------------------------
# Workspace functions
# ---------------------------------------------------------------------------

def _ensure_spy_workspace(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: list[int],
    min_obs: int | None = None,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Idempotent. If ALL workspace columns for the requested windows already
    exist, return panel unchanged. Otherwise run full/down/up univariate OLS
    per ticker for each window.

    Cached columns per window W:
      _ws_beta_{W}, _ws_alpha_{W}, _ws_residual_{W},
      _ws_downside_beta_{W}, _ws_upside_beta_{W}
    """
    _require_columns(panel, _REQUIRED_PANEL)
    if market_col not in market_returns.columns:
        raise ValueError(f"market_returns missing column: {market_col!r}")
    if "date" not in market_returns.columns:
        raise ValueError("market_returns missing column: 'date'")

    expected = _spy_ws_expected(windows)
    if all(c in panel.columns for c in expected):
        return panel

    if panel.empty:
        out = panel.copy()
        for col in expected:
            out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work["_log_ret"] = work.groupby("ticker", sort=False)["close"].transform(log_return)
    work = work.merge(
        market_returns[["date", market_col]],
        on="date",
        how="left",
    )

    for w in windows:
        beta_col = _ws_col("beta", w)
        alpha_col = _ws_col("alpha", w)
        resid_col = _ws_col("residual", w)
        down_col = _ws_col("downside_beta", w)
        up_col = _ws_col("upside_beta", w)

        if all(c in work.columns for c in (beta_col, alpha_col, resid_col, down_col, up_col)):
            continue

        work[beta_col] = np.nan
        work[alpha_col] = np.nan
        work[resid_col] = np.nan
        work[down_col] = np.nan
        work[up_col] = np.nan

        obs_floor = min_obs if min_obs is not None else max(20, w // 4)

        for _, grp in work.groupby("ticker", sort=False):
            y = grp["_log_ret"]
            x = grp[market_col]

            full = rolling_ols_stats(y, x, w)
            work.loc[grp.index, alpha_col] = full["alpha"].to_numpy()
            work.loc[grp.index, beta_col] = full["beta"].to_numpy()

            resid = rolling_residual(y, x, full["alpha"], full["beta"])
            work.loc[grp.index, resid_col] = resid.to_numpy()

            down = rolling_conditional_ols_stats(y, x, w, side="down", min_obs=obs_floor)
            work.loc[grp.index, down_col] = down["beta"].to_numpy()

            up = rolling_conditional_ols_stats(y, x, w, side="up", min_obs=obs_floor)
            work.loc[grp.index, up_col] = up["beta"].to_numpy()

    drop = ["_log_ret", market_col]
    work = work.drop(columns=[c for c in drop if c in work.columns])
    return _restore_order(work, original_index)


def _ensure_ff_workspace(
    panel: pd.DataFrame,
    ff_factors: pd.DataFrame,
    *,
    windows: list[int],
) -> pd.DataFrame:
    """
    Idempotent. Run 4-factor multivariate OLS per ticker for each window.

    Cached columns per window W:
      _ws_smart_beta_smb_{W}, _ws_smart_beta_hml_{W}, _ws_smart_beta_mom_{W},
      _ws_ff4_residual_{W}
    """
    _require_columns(panel, _REQUIRED_PANEL)
    ff_missing = _FF_REQUIRED_COLS - set(ff_factors.columns)
    if ff_missing:
        raise ValueError(f"ff_factors missing columns: {sorted(ff_missing)}")

    expected = _ff_ws_expected(windows)
    if all(c in panel.columns for c in expected):
        return panel

    if panel.empty:
        out = panel.copy()
        for col in expected:
            out[col] = pd.Series(dtype=float)
        return out

    original_index = panel.index
    work = _sorted_by_ticker_date(panel.copy())
    work["_log_ret"] = work.groupby("ticker", sort=False)["close"].transform(log_return)
    work = work.merge(
        ff_factors[["date", "mkt_rf", "smb", "hml", "mom", "rf"]],
        on="date",
        how="left",
    )
    work["_excess_ret"] = work["_log_ret"] - work["rf"]

    factor_cols = ["mkt_rf", "smb", "hml", "mom"]

    for w in windows:
        smb_col = _ws_col("smart_beta_smb", w)
        hml_col = _ws_col("smart_beta_hml", w)
        mom_col = _ws_col("smart_beta_mom", w)
        resid_col = _ws_col("ff4_residual", w)

        if all(c in work.columns for c in (smb_col, hml_col, mom_col, resid_col)):
            continue

        work[smb_col] = np.nan
        work[hml_col] = np.nan
        work[mom_col] = np.nan
        work[resid_col] = np.nan

        for _, grp in work.groupby("ticker", sort=False):
            y = grp["_excess_ret"]
            X = grp[factor_cols]
            stats = rolling_multi_ols_stats(y, X, w)

            work.loc[grp.index, smb_col] = stats["smb"].to_numpy()
            work.loc[grp.index, hml_col] = stats["hml"].to_numpy()
            work.loc[grp.index, mom_col] = stats["mom"].to_numpy()

            alpha_s = stats["alpha"].to_numpy()
            mkt_s = stats["mkt_rf"].to_numpy()
            smb_s = stats["smb"].to_numpy()
            hml_s = stats["hml"].to_numpy()
            mom_s = stats["mom"].to_numpy()

            y_arr = y.to_numpy(dtype=float)
            mkt_arr = grp["mkt_rf"].to_numpy(dtype=float)
            smb_arr = grp["smb"].to_numpy(dtype=float)
            hml_arr = grp["hml"].to_numpy(dtype=float)
            mom_arr = grp["mom"].to_numpy(dtype=float)

            resid = (
                y_arr - alpha_s
                - mkt_s * mkt_arr
                - smb_s * smb_arr
                - hml_s * hml_arr
                - mom_s * mom_arr
            )
            work.loc[grp.index, resid_col] = resid

    drop = ["_log_ret", "_excess_ret", "mkt_rf", "smb", "hml", "mom", "rf"]
    work = work.drop(columns=[c for c in drop if c in work.columns])
    return _restore_order(work, original_index)


# ---------------------------------------------------------------------------
# Cleanup utility
# ---------------------------------------------------------------------------

def drop_beta_workspace(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop all ``_ws_*`` workspace columns from the panel."""
    ws_cols = [c for c in panel.columns if c.startswith(_WS_PREFIX)]
    if ws_cols:
        return panel.drop(columns=ws_cols)
    return panel


# ---------------------------------------------------------------------------
# Column parser for Alphalens workflow
# ---------------------------------------------------------------------------

_DUAL_SUFFIX_RE = re.compile(r"^(.+?)_(\d+)_(\d+)$")
_SINGLE_SUFFIX_RE = re.compile(r"^(.+?)_(\d+)$")


def parse_beta_factor_name(col: str) -> dict | None:
    """
    Decode an H-004 factor column name into its parameters.

    Returns None if the column does not match any H-004 pattern.

    Examples
    --------
    >>> parse_beta_factor_name("beta_252")
    {'feature': 'beta', 'window': 252}
    >>> parse_beta_factor_name("residual_mom_252_21")
    {'feature': 'residual_mom', 'K': 252, 'S': 21}
    >>> parse_beta_factor_name("beta")
    {'feature': 'beta', 'window': None}
    >>> parse_beta_factor_name("obv_mom_signed")  # not H-004
    """
    if col in _SINGLE_WINDOW_FEATURES:
        return {"feature": col, "window": None}
    if col in _DUAL_WINDOW_FEATURES:
        return {"feature": col, "K": None, "S": None}

    for stem in sorted(_DUAL_WINDOW_FEATURES, key=lambda s: -len(s)):
        prefix = stem + "_"
        if col.startswith(prefix):
            rest = col[len(prefix):]
            parts = rest.split("_")
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                return {"feature": stem, "K": int(parts[0]), "S": int(parts[1])}

    for stem in sorted(_SINGLE_WINDOW_FEATURES, key=lambda s: -len(s)):
        prefix = stem + "_"
        if col.startswith(prefix):
            rest = col[len(prefix):]
            if rest.isdigit():
                return {"feature": stem, "window": int(rest)}

    return None
