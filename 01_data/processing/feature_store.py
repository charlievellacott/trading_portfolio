"""Feature-store entrypoints that add alpha columns to long OHLCV panels."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import pandas as pd

from data.processing.feature_implementation.beta import (
    blume_adjust,
    normalize_windows,
    regression_column_name,
    residual_momentum_signal,
    windowed_column_name,
)
from data.processing.feature_implementation.beta_features import (
    _ensure_ff_workspace,
    _ensure_spy_workspace,
    _ws_col,
    drop_beta_workspace,
    parse_beta_factor_name,
)
from data.processing.feature_implementation.gk_vol_ratio import (
    VALID_MODES as GK_VALID_MODES,
    add_gk_realised_ratio_raw,
    apply_ratio_mode,
)
from data.processing.feature_implementation.idiosyncratic_vol import (
    add_idiosyncratic_vol as add_idiosyncratic_vol_raw,
)
from data.processing.feature_implementation.obv_momentum import (
    VALID_MODES,
    add_obv_confirmed_combined,
    cross_sectional_pct_rank,
)

WindowSpec = int | list[int] | tuple[int, ...] | Sequence[int]


def _normalize_nonneg_windows(windows: WindowSpec, *, name: str) -> list[int]:
    """Like ``normalize_windows`` but allow 0 (for momentum skip)."""
    if isinstance(windows, bool):
        raise ValueError(f"{name} must be a non-negative int or a list of non-negative ints")
    if isinstance(windows, int):
        items = [windows]
    elif isinstance(windows, (list, tuple)):
        items = list(windows)
    else:
        raise ValueError(f"{name} must be a non-negative int or a list of non-negative ints")
    if not items:
        raise ValueError(f"{name} must be a non-empty list of non-negative ints")
    for w in items:
        if not isinstance(w, int) or isinstance(w, bool) or w < 0:
            raise ValueError(f"{name} entries must be non-negative ints, got {w!r}")
    return items


def add_obv_confirmed_momentum(
    panel: pd.DataFrame,
    *,
    lookback: WindowSpec = 252,
    skip: WindowSpec = 21,
    obv_window: WindowSpec = 20,
    mode: str = "signed",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add H-001 OBV-confirmed momentum column(s) ``obv_mom_{mode}`` [``_{L}_{S}_{W}``].

    Features at date ``t`` use OHLCV through the close of ``t``. The intended
    prediction target is the next close (``P_{t+1} / P_t - 1``); labels are not
    added here.

    When ``lookback``, ``skip``, and ``obv_window`` each resolve to a single
    value (one combo), the column is ``obv_mom_{mode}``. When any kwarg is a
    list yielding more than one ``(L, S, W)`` combo, columns are
    ``obv_mom_{mode}_{L}_{S}_{W}``.

    Parameters
    ----------
    panel:
        Long-format frame with ``date``, ``ticker``, ``close``, ``volume``.
    lookback, skip:
        Momentum windows (L, S): ``P_{t-S} / P_{t-L} - 1``. Each may be an int
        or a list of ints (cartesian product with ``obv_window``).
    obv_window:
        OBV trend window W: ``OBV_t - OBV_{t-W}``. Int or list of ints.
    mode:
        ``"signed"`` (default), ``"strict_zero"``, or ``"soft"``.
    normalize:
        If True, store cross-sectional percentile rank of the combined signal
        within each date (GBM-ready), applied per output column. If False,
        store the raw combined signal.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    lookbacks = normalize_windows(lookback)
    skips = _normalize_nonneg_windows(skip, name="skip")
    obv_windows = normalize_windows(obv_window)
    combos = list(itertools.product(lookbacks, skips, obv_windows))
    for L, S, W in combos:
        if L <= S:
            raise ValueError(
                f"lookback must be greater than skip for every combo, got L={L}, S={S}"
            )

    required = {"date", "ticker", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    stem = f"obv_mom_{mode}"
    multi = len(combos) > 1
    out_cols = [windowed_column_name(stem, L, S, W, multi=multi) for L, S, W in combos]

    if panel.empty:
        out = panel.copy()
        for col in out_cols:
            out[col] = pd.Series(dtype=float)
        return out

    result = panel.copy()
    for (L, S, W), out_col in zip(combos, out_cols):
        combined_col = f"_obv_confirmed_combined_tmp_{L}_{S}_{W}"
        result = add_obv_confirmed_combined(
            result,
            lookback=L,
            skip=S,
            obv_window=W,
            mode=mode,
            col=combined_col,
        )
        if normalize:
            result[out_col] = cross_sectional_pct_rank(result, combined_col)
        else:
            result[out_col] = result[combined_col]
        result = result.drop(columns=[combined_col])

    return result


def add_gk_vol_ratio(
    panel: pd.DataFrame,
    *,
    gk_window: WindowSpec = 5,
    realised_window: WindowSpec = 20,
    mode: str = "ratio",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add H-002 GK / realised vol ratio column(s) ``gk_vol_{mode}`` [``_{gkW}_{realW}``].

    Features at date ``t`` use OHLC through the close of ``t``. The intended
    prediction target is the next close (``P_{t+1} / P_t - 1``); labels are not
    added here. No denominator floor and no winsorization are applied.

    When ``gk_window`` and ``realised_window`` yield one combo, the column is
    ``gk_vol_{mode}``. When more than one combo, columns are
    ``gk_vol_{mode}_{gkW}_{realW}``.

    Parameters
    ----------
    panel:
        Long-format frame with ``date``, ``ticker``, ``open``, ``high``,
        ``low``, ``close``.
    gk_window:
        Short window for the mean of daily Garman–Klass volatility. Int or list.
    realised_window:
        Number of log close-to-close returns in the realised-vol std
        (ending at ``t``). Int or list.
    mode:
        ``"ratio"`` (default), ``"log_ratio"``, or ``"reversal"``.
    normalize:
        If True, store cross-sectional percentile rank of the mode-transformed
        signal within each date (GBM-ready), applied per output column. If False,
        store the unranked value.
    """
    if mode not in GK_VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(GK_VALID_MODES)}, got {mode!r}")

    gk_windows = normalize_windows(gk_window)
    realised_windows = normalize_windows(realised_window)
    combos = list(itertools.product(gk_windows, realised_windows))

    required = {"date", "ticker", "open", "high", "low", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    stem = f"gk_vol_{mode}"
    multi = len(combos) > 1
    out_cols = [
        windowed_column_name(stem, gk_w, real_w, multi=multi) for gk_w, real_w in combos
    ]

    if panel.empty:
        out = panel.copy()
        for col in out_cols:
            out[col] = pd.Series(dtype=float)
        return out

    result = panel.copy()
    for (gk_w, real_w), out_col in zip(combos, out_cols):
        raw_col = f"_gk_realised_ratio_raw_tmp_{gk_w}_{real_w}"
        mode_col = f"_gk_vol_mode_tmp_{gk_w}_{real_w}"
        result = add_gk_realised_ratio_raw(
            result,
            gk_window=gk_w,
            realised_window=real_w,
            col=raw_col,
        )
        result[mode_col] = apply_ratio_mode(result[raw_col], mode=mode)
        if normalize:
            result[out_col] = cross_sectional_pct_rank(result, mode_col)
        else:
            result[out_col] = result[mode_col]
        result = result.drop(columns=[raw_col, mode_col])

    return result


def add_idiosyncratic_vol(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: WindowSpec = 20,
    normalize: bool = True,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-003 idiosyncratic-vol column(s) ``idio_vol`` [``_{w}``].

    Features at date ``t`` use stock and market log returns through the close
    of ``t``. The intended prediction target is the next close
    (``P_{t+1} / P_t - 1``); labels are not added here. The benchmark is
    whatever series is supplied in ``market_returns`` (research notebooks lock
    this to SPY).

    One window → ``idio_vol``; multiple windows → ``idio_vol_{w}``.

    Parameters
    ----------
    panel:
        Long-format frame with ``date``, ``ticker``, ``close``.
    market_returns:
        Frame with ``date`` and ``market_col`` (from ``market_return_frame``).
    windows:
        Rolling OLS / residual-std window length(s).
    normalize:
        If True, store cross-sectional percentile rank of raw idio vol within
        each date (GBM-ready). If False, store raw residual std (``ddof=1``).
    market_col:
        Column name in ``market_returns`` for benchmark log returns.
    """
    window_list = normalize_windows(windows)
    multi = len(window_list) > 1
    out_cols = [
        regression_column_name("idio_vol", w, multi_window=multi) for w in window_list
    ]

    required = {"date", "ticker", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    if market_col not in market_returns.columns:
        raise ValueError(f"market_returns missing column: {market_col!r}")
    if "date" not in market_returns.columns:
        raise ValueError("market_returns missing column: 'date'")

    if panel.empty:
        out = panel.copy()
        for col in out_cols:
            out[col] = pd.Series(dtype=float)
        return out

    result = add_idiosyncratic_vol_raw(
        panel,
        market_returns,
        windows=window_list if multi else window_list[0],
        market_col=market_col,
    )
    if normalize:
        for col in out_cols:
            result[col] = cross_sectional_pct_rank(result, col)
    return result


# ---------------------------------------------------------------------------
# H-004 · Beta Feature Suite store callers
# ---------------------------------------------------------------------------


def add_beta(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    benchmark: str = "spy",
    windows: WindowSpec = 252,
    normalize: bool = True,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 beta column(s).

    ``benchmark='spy'``: univariate β vs SPY → column(s) ``beta`` / ``beta_{W}``.
    ``benchmark='ff'``: 4-factor loadings → ``smart_beta_smb/hml/mom`` [``_{W}``].

    Pass ``windows`` as a list for multi-window Alphalens screening.
    """
    if benchmark not in ("spy", "ff"):
        raise ValueError(f"benchmark must be 'spy' or 'ff', got {benchmark!r}")

    window_list = normalize_windows(windows)
    multi = len(window_list) > 1

    if benchmark == "spy":
        result = _ensure_spy_workspace(
            panel, factors, windows=window_list, market_col=market_col,
        )
        for w in window_list:
            ws = _ws_col("beta", w)
            out = regression_column_name("beta", w, multi_window=multi)
            result[out] = result[ws]
            if normalize:
                result[out] = cross_sectional_pct_rank(result, out)
    else:
        result = _ensure_ff_workspace(panel, factors, windows=window_list)
        for stem in ("smart_beta_smb", "smart_beta_hml", "smart_beta_mom"):
            for w in window_list:
                ws = _ws_col(stem, w)
                out = regression_column_name(stem, w, multi_window=multi)
                result[out] = result[ws]
                if normalize:
                    result[out] = cross_sectional_pct_rank(result, out)
    return result


def add_downside_beta(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: WindowSpec = 252,
    normalize: bool = True,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 downside-beta column(s) ``downside_beta`` [``_{W}``].

    Threshold: in-window mean of market returns. min_obs: max(20, w // 4).
    """
    window_list = normalize_windows(windows)
    multi = len(window_list) > 1

    result = _ensure_spy_workspace(
        panel, market_returns, windows=window_list, market_col=market_col,
    )
    for w in window_list:
        ws = _ws_col("downside_beta", w)
        out = regression_column_name("downside_beta", w, multi_window=multi)
        result[out] = result[ws]
        if normalize:
            result[out] = cross_sectional_pct_rank(result, out)
    return result


def add_upside_beta(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: WindowSpec = 252,
    normalize: bool = True,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 upside-beta column(s) ``upside_beta`` [``_{W}``].

    Threshold: in-window mean of market returns. min_obs: max(20, w // 4).
    """
    window_list = normalize_windows(windows)
    multi = len(window_list) > 1

    result = _ensure_spy_workspace(
        panel, market_returns, windows=window_list, market_col=market_col,
    )
    for w in window_list:
        ws = _ws_col("upside_beta", w)
        out = regression_column_name("upside_beta", w, multi_window=multi)
        result[out] = result[ws]
        if normalize:
            result[out] = cross_sectional_pct_rank(result, out)
    return result


def add_net_beta_spread(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: WindowSpec = 252,
    normalize: bool = True,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 net beta-spread column(s) ``net_beta_spread`` [``_{W}``].

    Defined as ``upside_beta - downside_beta`` for each window.
    """
    window_list = normalize_windows(windows)
    multi = len(window_list) > 1

    result = _ensure_spy_workspace(
        panel, market_returns, windows=window_list, market_col=market_col,
    )
    for w in window_list:
        up_ws = _ws_col("upside_beta", w)
        down_ws = _ws_col("downside_beta", w)
        out = regression_column_name("net_beta_spread", w, multi_window=multi)
        result[out] = result[up_ws] - result[down_ws]
        if normalize:
            result[out] = cross_sectional_pct_rank(result, out)
    return result


def add_relative_downside_beta(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: WindowSpec = 252,
    normalize: bool = True,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 relative downside-beta column(s) ``rel_downside_beta`` [``_{W}``].

    Defined as ``downside_beta - beta`` for each window.
    """
    window_list = normalize_windows(windows)
    multi = len(window_list) > 1

    result = _ensure_spy_workspace(
        panel, market_returns, windows=window_list, market_col=market_col,
    )
    for w in window_list:
        down_ws = _ws_col("downside_beta", w)
        beta_ws = _ws_col("beta", w)
        out = regression_column_name("rel_downside_beta", w, multi_window=multi)
        result[out] = result[down_ws] - result[beta_ws]
        if normalize:
            result[out] = cross_sectional_pct_rank(result, out)
    return result


def add_relative_upside_beta(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: WindowSpec = 252,
    normalize: bool = True,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 relative upside-beta column(s) ``rel_upside_beta`` [``_{W}``].

    Defined as ``upside_beta - beta`` for each window.
    """
    window_list = normalize_windows(windows)
    multi = len(window_list) > 1

    result = _ensure_spy_workspace(
        panel, market_returns, windows=window_list, market_col=market_col,
    )
    for w in window_list:
        up_ws = _ws_col("upside_beta", w)
        beta_ws = _ws_col("beta", w)
        out = regression_column_name("rel_upside_beta", w, multi_window=multi)
        result[out] = result[up_ws] - result[beta_ws]
        if normalize:
            result[out] = cross_sectional_pct_rank(result, out)
    return result


def add_blume_beta(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    windows: WindowSpec = 252,
    normalize: bool = False,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 Blume-adjusted beta column(s) ``blume_beta`` [``_{W}``].

    Formula: ``0.67 * beta + 0.33``. Default ``normalize=False``.
    """
    window_list = normalize_windows(windows)
    multi = len(window_list) > 1

    result = _ensure_spy_workspace(
        panel, market_returns, windows=window_list, market_col=market_col,
    )
    for w in window_list:
        ws = _ws_col("beta", w)
        out = regression_column_name("blume_beta", w, multi_window=multi)
        result[out] = blume_adjust(result[ws])
        if normalize:
            result[out] = cross_sectional_pct_rank(result, out)
    return result


def add_residual_momentum(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    benchmark: str = "spy",
    formation_window: WindowSpec = 252,
    skip: WindowSpec = 21,
    normalize: bool = False,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 residual-momentum column(s).

    ``benchmark='spy'``: CAPM residuals → ``residual_mom`` [``_{K}_{S}``].
    ``benchmark='ff'``: 4-factor residuals → ``smart_residual_mom`` [``_{K}_{S}``].

    Cartesian product of ``formation_window × skip`` → one column per combo.
    ``formation_window`` values must be present in the workspace's cached windows.
    ``skip`` does NOT require a separate OLS.
    """
    if benchmark not in ("spy", "ff"):
        raise ValueError(f"benchmark must be 'spy' or 'ff', got {benchmark!r}")

    formation_list = normalize_windows(formation_window)
    skip_list = _normalize_nonneg_windows(skip, name="skip")

    for k in formation_list:
        for s in skip_list:
            if k <= s:
                raise ValueError(
                    f"formation_window must be > skip for every combo, got K={k}, S={s}"
                )

    combos = list(itertools.product(formation_list, skip_list))
    stem = "residual_mom" if benchmark == "spy" else "smart_residual_mom"
    multi = len(combos) > 1

    if benchmark == "spy":
        result = _ensure_spy_workspace(
            panel, factors, windows=formation_list, market_col=market_col,
        )
        resid_prefix = "residual"
    else:
        result = _ensure_ff_workspace(panel, factors, windows=formation_list)
        resid_prefix = "ff4_residual"

    for k, s in combos:
        ws = _ws_col(resid_prefix, k)
        out = windowed_column_name(stem, k, s, multi=multi)

        # Compute per-ticker residual momentum signal
        signals = []
        for _, grp in result.groupby("ticker", sort=False):
            sig = residual_momentum_signal(grp[ws], k, s)
            signals.append(sig)
        combined = pd.concat(signals)
        result[out] = combined.reindex(result.index)

        if normalize:
            result[out] = cross_sectional_pct_rank(result, out)

    return result
