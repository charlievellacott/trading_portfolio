"""Feature-store entrypoints that add alpha columns to long OHLCV panels."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from data.processing.feature_implementation.beta_features import (
    _ensure_ff_workspace,
    _ensure_spy_workspace,
    _ws_col,
    add_beta_mkt_interact,
    add_market_corr,
    add_mkt_near_52w,
    add_mkt_ret,
    add_mkt_vol,
    add_r_squared,
    add_rel_strength,
    blume_adjust,
    drop_beta_workspace,
    parse_beta_factor_name,
    residual_momentum_signal,
)
from data.processing.feature_implementation.gk_vol_ratio import (
    VALID_MODES as GK_VALID_MODES,
    add_gk_realised_ratio_raw,
    apply_ratio_mode,
)
from data.processing.feature_implementation.gross_profitability import (
    add_gross_profitability as _add_gross_profitability_raw,
)
from data.processing.feature_implementation.idiosyncratic_vol import (
    add_idiosyncratic_vol as add_idiosyncratic_vol_raw,
)
from data.processing.feature_implementation.momentum import (
    VALID_MAX_LOTTERY_MODES,
    VALID_NEAR_52W_MODES,
    add_max_lottery_raw,
    add_near_52w_raw,
    apply_near_52w_mode,
)
from data.processing.feature_implementation.obv_momentum import (
    VALID_MODES,
    add_obv_confirmed_combined,
)
from data.processing.feature_implementation.size_and_valuation_features import (
    add_amihud as _add_amihud_raw,
    add_book_yield as _add_book_yield_raw,
    add_earnings_yield as _add_earnings_yield_raw,
    add_log_market_cap as _add_log_mcap_raw,
    add_size_momentum as _add_size_mom_raw,
    add_valuation_roc as _add_val_roc_raw,
    add_value_momentum_distance as _add_val_mom_dist_raw,
    add_value_momentum_interaction as _add_val_mom_interact_raw,
    add_value_momentum_residual as _add_val_mom_resid_raw,
)
from data.processing.feature_implementation.short_flow import (
    add_abnormal_short_flow as _add_abnormal_short_flow_raw,
    add_short_volume_ratio as _add_short_volume_ratio_raw,
    short_volume_ratio,
)
from data.processing.feature_implementation.filing_clock import (
    add_days_since_filing as _add_days_since_filing_raw,
    add_expected_days_until_filing as _add_expected_days_until_filing_raw,
)
from data.processing.feature_implementation.utilities import (
    cross_sectional_ols_residual,
    cross_sectional_pct_rank,
    cross_sectional_zscore,
    normalize_windows,
    regression_column_name,
    windowed_column_name,
)

WindowSpec = int | list[int] | tuple[int, ...] | Sequence[int]

_VALID_VAL_METRICS = frozenset({"pe", "pb"})
_VALID_SHORT_FLOW_MODES = frozenset({"abnormal", "ratio", "exempt_ratio"})
_VALID_FILING_CLOCK_MODES = frozenset({"since", "expected_until"})


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


# ---------------------------------------------------------------------------
# H-001 · OBV-confirmed momentum
# ---------------------------------------------------------------------------


def add_obv_confirmed_momentum(
    panel: pd.DataFrame,
    *,
    lookback: WindowSpec = 252,
    skip: WindowSpec = 21,
    obv_window: WindowSpec = 20,
    mode: str = "signed",
) -> pd.DataFrame:
    """
    Add H-001 OBV-confirmed momentum column(s) ``obv_mom_{mode}`` [``_{L}_{S}_{W}``].

    Features at date ``t`` use OHLCV through the close of ``t``. The intended
    prediction target is the next close (``P_{t+1} / P_t - 1``); labels are not
    added here. The store always writes the raw combined signal (return-like;
    magnitude retained) — there is no ``normalize`` kwarg. Soft mode still uses
    an internal CS pct-rank of OBV trend as a weight inside the combined signal.

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
        or a list of ints (cartesian product with ``obv_window``). Pairs with
        ``L <= S`` are skipped; if none remain, raises ``ValueError``.
    obv_window:
        OBV trend window W: ``OBV_t - OBV_{t-W}``. Int or list of ints.
    mode:
        ``"signed"`` (default), ``"strict_zero"``, or ``"soft"``.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    lookbacks = normalize_windows(lookback)
    skips = _normalize_nonneg_windows(skip, name="skip")
    obv_windows = normalize_windows(obv_window)
    combos = [
        (L, S, W)
        for L, S, W in itertools.product(lookbacks, skips, obv_windows)
        if L > S
    ]
    if not combos:
        raise ValueError(
            "lookback must be greater than skip for every combo; "
            f"no valid combos from lookback={list(lookbacks)}, skip={list(skips)}"
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
        result[out_col] = result[combined_col]
        result = result.drop(columns=[combined_col])

    return result


# ---------------------------------------------------------------------------
# H-006 · 52-week high proximity
# ---------------------------------------------------------------------------


def add_near_52w(
    panel: pd.DataFrame,
    *,
    window: WindowSpec = 252,
    mode: str = "ratio",
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Add H-006 near-52-week-high column(s) ``near_52w_{mode}`` [``_{W}``].

    Features at date ``t`` use ``close`` and ``high`` through the close of
    ``t`` (today is included in the rolling peak). The intended prediction
    target is the next close (``P_{t+1} / P_t - 1``); labels are not added
    here. No denominator floor and no winsorization are applied.

    When ``window`` is a single int, the column is ``near_52w_{mode}``. When
    ``window`` is a list with more than one value, columns are
    ``near_52w_{mode}_{W}``.

    Parameters
    ----------
    panel:
        Long-format frame with ``date``, ``ticker``, ``close``, ``high``.
    window:
        Trading-day lookback for the rolling peak (default 252 ≈ 52 weeks).
        Int or list of ints.
    mode:
        ``"ratio"`` (default): ``close / Hmax``.
        ``"log_drawdown"``: ``ln(close / Hmax)`` when the ratio is positive.
    normalize:
        If True, store cross-sectional percentile rank of the mode-transformed
        signal within each date (CS-aligned for pooled ranking / Alphalens).
        If False (default), store the unranked value (ratio is already unitless
        and roughly in ``(0, 1]``).
    """
    if mode not in VALID_NEAR_52W_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALID_NEAR_52W_MODES)}, got {mode!r}"
        )

    window_list = normalize_windows(window)
    multi = len(window_list) > 1

    required = {"date", "ticker", "close", "high"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    stem = f"near_52w_{mode}"
    out_cols = [windowed_column_name(stem, w, multi=multi) for w in window_list]

    if panel.empty:
        out = panel.copy()
        for col in out_cols:
            out[col] = pd.Series(dtype=float)
        return out

    result = panel.copy()
    for w, out_col in zip(window_list, out_cols):
        raw_col = f"_near_52w_raw_tmp_{w}"
        mode_col = f"_near_52w_mode_tmp_{w}"
        result = add_near_52w_raw(result, window=w, col=raw_col)
        result[mode_col] = apply_near_52w_mode(result[raw_col], mode=mode)
        if normalize:
            result[out_col] = cross_sectional_pct_rank(result, mode_col)
        else:
            result[out_col] = result[mode_col]
        result = result.drop(columns=[raw_col, mode_col])

    return result


# ---------------------------------------------------------------------------
# H-007 · MAX (lottery demand)
# ---------------------------------------------------------------------------


def add_max_lottery(
    panel: pd.DataFrame,
    *,
    n_extreme: WindowSpec = 5,
    window: WindowSpec = 21,
    mode: str = "simple",
    normalize: bool = True,
    add_residuals: bool = False,
    idio_vol_col: str = "idio_vol",
) -> pd.DataFrame:
    """
    Add H-007 MAX lottery column(s) ``max_lottery_{mode}`` [``_{N}_{W}``].

    Features at date ``t`` use ``close`` through the close of ``t`` (daily
    returns ending at ``t`` are included in the extreme window). The intended
    prediction target is the next close (``P_{t+1} / P_t - 1``); labels are
    not added here. No denominator floor and no winsorization are applied.

    When ``n_extreme`` and ``window`` yield one combo, the column is
    ``max_lottery_{mode}``. When more than one combo, columns are
    ``max_lottery_{mode}_{N}_{W}``. With ``add_residuals=True``, also writes
    ``max_lottery_{mode}_resid`` [same suffix] = within-date OLS residual of
    CS-rank(MAX) on CS-rank(``idio_vol_col``). Residuals require that column
    already on the panel (caller must run ``add_idiosyncratic_vol`` first).

    Parameters
    ----------
    panel:
        Long-format frame with ``date``, ``ticker``, ``close``.
    n_extreme:
        Number of largest daily returns to average (default 5). Int or list.
    window:
        Trailing trading-day return window (default 21). Int or list.
    mode:
        ``"simple"`` (default): ``P_t / P_{t-1} - 1``.
        ``"log"``: ``ln(P_t / P_{t-1})``.
    normalize:
        If True (default), store cross-sectional z-score of MAX within each
        date (preserves lottery extremity better than pct-rank). If False,
        store raw MAX. No sign flip.
    add_residuals:
        If True, also store the within-date residual of CS-rank(MAX) on
        CS-rank(``idio_vol_col``). Residuals always use CS ranks even when
        ``normalize=False``.
    idio_vol_col:
        Control column for residuals (default ``"idio_vol"``).
    """
    if mode not in VALID_MAX_LOTTERY_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALID_MAX_LOTTERY_MODES)}, got {mode!r}"
        )

    n_list = normalize_windows(n_extreme)
    w_list = normalize_windows(window)
    combos = list(itertools.product(n_list, w_list))
    for n, w in combos:
        if w < n:
            raise ValueError(
                f"window must be >= n_extreme for every combo; got n_extreme={n}, window={w}"
            )
    multi = len(combos) > 1

    required = {"date", "ticker", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    if add_residuals and idio_vol_col not in panel.columns:
        raise ValueError(
            f"add_residuals=True requires column {idio_vol_col!r} on the panel "
            "(run add_idiosyncratic_vol first)"
        )

    stem = f"max_lottery_{mode}"
    resid_stem = f"max_lottery_{mode}_resid"
    out_cols = [windowed_column_name(stem, n, w, multi=multi) for n, w in combos]
    resid_cols = (
        [windowed_column_name(resid_stem, n, w, multi=multi) for n, w in combos]
        if add_residuals
        else []
    )

    if panel.empty:
        out = panel.copy()
        for col in out_cols + resid_cols:
            out[col] = pd.Series(dtype=float)
        return out

    result = panel.copy()
    for (n, w), out_col in zip(combos, out_cols):
        raw_col = f"_max_lottery_raw_tmp_{n}_{w}"
        result = add_max_lottery_raw(
            result, n_extreme=n, window=w, mode=mode, col=raw_col
        )
        if normalize:
            result[out_col] = cross_sectional_zscore(result, raw_col)
        else:
            result[out_col] = result[raw_col]

        if add_residuals:
            resid_col = windowed_column_name(resid_stem, n, w, multi=multi)
            y_rank_col = f"_max_y_rank_tmp_{n}_{w}"
            x_rank_col = f"_max_x_rank_tmp_{n}_{w}"
            result[y_rank_col] = cross_sectional_pct_rank(result, raw_col)
            result[x_rank_col] = cross_sectional_pct_rank(result, idio_vol_col)
            result[resid_col] = cross_sectional_ols_residual(
                result, y_rank_col, x_rank_col
            )
            result = result.drop(columns=[y_rank_col, x_rank_col])

        result = result.drop(columns=[raw_col])

    return result


# ---------------------------------------------------------------------------
# H-002 · GK vol ratio
# ---------------------------------------------------------------------------


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
        Short window for the mean of daily Garman-Klass volatility. Int or list.
    realised_window:
        Number of log close-to-close returns in the realised-vol std
        (ending at ``t``). Int or list.
    mode:
        ``"ratio"`` (default), ``"log_ratio"``, or ``"reversal"``.
    normalize:
        If True (default), store cross-sectional z-score of the mode-transformed
        signal within each date (preserves stress magnitude better than
        pct-rank). If False, store the unranked value.
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
            result[out_col] = cross_sectional_zscore(result, mode_col)
        else:
            result[out_col] = result[mode_col]
        result = result.drop(columns=[raw_col, mode_col])

    return result


# ---------------------------------------------------------------------------
# H-003 · Idiosyncratic volatility
# ---------------------------------------------------------------------------


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
    whatever series is supplied in ``market_returns`` (typically SPY or RSP
    via ``fetch_ohlcv`` + ``market_return_frame``).

    One window -> ``idio_vol``; multiple windows -> ``idio_vol_{w}``.

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
        each date (CS-aligned for pooled ranking / Alphalens; the factor is
        the IVOL rank in the lit). If False, store raw residual std (``ddof=1``).
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

    ``benchmark='spy'``: univariate beta vs SPY -> column(s) ``beta`` / ``beta_{W}``.
    ``benchmark='rsp'``: univariate beta vs RSP (equal-weight S&P) -> same ``beta`` columns.
    ``benchmark='ff'``: 4-factor loadings -> ``smart_beta_smb/hml/mom`` [``_{W}``].

    For ``spy`` / ``rsp``, pass ``market_returns`` from ``fetch_ohlcv`` +
    ``market_return_frame`` (ticker must match the flag). Store does not fetch.

    Pass ``windows`` as a list for multi-window Alphalens screening.
    """
    if benchmark not in ("spy", "rsp", "ff"):
        raise ValueError(
            f"benchmark must be 'spy', 'rsp', or 'ff', got {benchmark!r}"
        )

    window_list = normalize_windows(windows)
    multi = len(window_list) > 1

    if benchmark in ("spy", "rsp"):
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
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 Blume-adjusted beta column(s) ``blume_beta`` [``_{W}``].

    Formula: ``0.67 * beta + 0.33``. Output is never CS-ranked (``normalize``
    is not offered — ranking would discard Blume shrinkage magnitude).
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
    return result


def add_residual_momentum(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    benchmark: str = "spy",
    formation_window: WindowSpec = 252,
    skip: WindowSpec = 21,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """
    Add H-004 residual-momentum column(s).

    ``benchmark='spy'``: CAPM residuals vs SPY -> ``residual_mom`` [``_{K}_{S}``].
    ``benchmark='rsp'``: CAPM residuals vs RSP -> same ``residual_mom`` columns.
    ``benchmark='ff'``: 4-factor residuals -> ``smart_residual_mom`` [``_{K}_{S}``].

    For ``spy`` / ``rsp``, pass ``market_returns`` from ``fetch_ohlcv`` +
    ``market_return_frame`` (ticker must match the flag). Store does not fetch.

    Cartesian product of ``formation_window x skip`` -> one column per combo.
    ``formation_window`` values must be present in the workspace's cached windows.
    ``skip`` does NOT require a separate OLS.

    Output is never CS-ranked (``normalize`` is not offered — ranking would
    discard residual-momentum magnitude).
    """
    if benchmark not in ("spy", "rsp", "ff"):
        raise ValueError(
            f"benchmark must be 'spy', 'rsp', or 'ff', got {benchmark!r}"
        )

    formation_list = normalize_windows(formation_window)
    skip_list = _normalize_nonneg_windows(skip, name="skip")

    for k in formation_list:
        for s in skip_list:
            if k <= s:
                raise ValueError(
                    f"formation_window must be > skip for every combo, got K={k}, S={s}"
                )

    combos = list(itertools.product(formation_list, skip_list))
    stem = (
        "residual_mom" if benchmark in ("spy", "rsp") else "smart_residual_mom"
    )
    multi = len(combos) > 1

    if benchmark in ("spy", "rsp"):
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

        signals = []
        for _, grp in result.groupby("ticker", sort=False):
            sig = residual_momentum_signal(grp[ws], k, s)
            signals.append(sig)
        combined = pd.concat(signals)
        result[out] = combined.reindex(result.index)

    return result


# ---------------------------------------------------------------------------
# H-008 · Gross Profitability
# ---------------------------------------------------------------------------


def add_gross_profitability(
    panel: pd.DataFrame,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add H-008 gross-profitability column ``gross_profitability``.

    Features at date ``t`` use fundamentals known by filing date ``<= t``
    (as ``gp_asset`` on the panel). Prefer
    ``add_gross_profitability_factors`` (fetches SEC GP unless
    ``gross_profitability_data_exists=True``); labels are not added here.
    ``gp_asset = gross_profit_ttm / assets``; NaN when assets missing or
    ``<= 0``. No floor and no winsorize in the store.

    If ``normalize=True`` (default), store the cross-sectional percentile
    rank of ``gp_asset`` within each date.
    """
    _col = "gross_profitability"
    required = {"date", "ticker", "gp_asset"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    result = _add_gross_profitability_raw(panel, col=_col)
    if normalize:
        result[_col] = cross_sectional_pct_rank(result, _col)
    return result


# ---------------------------------------------------------------------------
# H-005 · Size & Value store callers
# ---------------------------------------------------------------------------


def add_book_yield(
    panel: pd.DataFrame,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add H-005 book-yield column ``book_yield``.

    Features at date ``t`` use data through ``t``; labels are not added here.
    ``book_yield = 1 / pb``; NaN when ``pb <= 0``.
    """
    _col = "book_yield"
    required = {"date", "ticker", "pb"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    result = _add_book_yield_raw(panel, col=_col)
    if normalize:
        result[_col] = cross_sectional_pct_rank(result, _col)
    return result


def add_earnings_yield(
    panel: pd.DataFrame,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add H-005 earnings-yield column ``earnings_yield``.

    Features at date ``t`` use data through ``t``; labels are not added here.
    ``earnings_yield = 1 / pe``; NaN when ``pe <= 0``.
    """
    _col = "earnings_yield"
    required = {"date", "ticker", "pe"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    result = _add_earnings_yield_raw(panel, col=_col)
    if normalize:
        result[_col] = cross_sectional_pct_rank(result, _col)
    return result


def add_log_mcap(
    panel: pd.DataFrame,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add H-005 log-market-cap column ``log_mcap``.

    Features at date ``t`` use data through ``t``; labels are not added here.
    ``log_mcap = log(market_cap)``; NaN when ``market_cap <= 0``.
    """
    _col = "log_mcap"
    required = {"date", "ticker", "market_cap"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    result = _add_log_mcap_raw(panel, col=_col)
    if normalize:
        result[_col] = cross_sectional_pct_rank(result, _col)
    return result


def add_valuation_roc(
    panel: pd.DataFrame,
    *,
    metric: str = "pb",
    window: WindowSpec = 63,
) -> pd.DataFrame:
    """
    Add H-005 valuation rate-of-change column(s) ``val_roc_{metric}`` [``_{W}``].

    Features at date ``t`` use data through ``t``; labels are not added here.
    Formula: ``log(val_t) - log(val_{t-L})``. Already return-like / ~stationary;
    output is never CS-ranked (``normalize`` is not offered).

    One window -> ``val_roc_{metric}``; multiple windows -> ``val_roc_{metric}_{W}``.
    """
    if metric not in _VALID_VAL_METRICS:
        raise ValueError(f"metric must be one of {sorted(_VALID_VAL_METRICS)}, got {metric!r}")

    window_list = normalize_windows(window)
    multi = len(window_list) > 1

    required = {"date", "ticker", metric}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    stem = f"val_roc_{metric}"
    out_cols = [
        regression_column_name(stem, w, multi_window=multi) for w in window_list
    ]

    if panel.empty:
        out = panel.copy()
        for col in out_cols:
            out[col] = pd.Series(dtype=float)
        return out

    result = panel.copy()
    for w, out_col in zip(window_list, out_cols):
        tmp_col = f"_val_roc_tmp_{metric}_{w}"
        result = _add_val_roc_raw(result, metric=metric, window=w, col=tmp_col)
        result[out_col] = result[tmp_col]
        result = result.drop(columns=[tmp_col])

    return result


def add_size_momentum(
    panel: pd.DataFrame,
    *,
    window: WindowSpec = 63,
) -> pd.DataFrame:
    """
    Add H-005 size-momentum column(s) ``size_mom`` [``_{W}``].

    Features at date ``t`` use data through ``t``; labels are not added here.
    Formula: ``log(mcap_t / mcap_{t-L})``. Already return-like / ~stationary;
    output is never CS-ranked (``normalize`` is not offered).

    One window -> ``size_mom``; multiple windows -> ``size_mom_{W}``.
    """
    window_list = normalize_windows(window)
    multi = len(window_list) > 1

    required = {"date", "ticker", "market_cap"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    out_cols = [
        regression_column_name("size_mom", w, multi_window=multi) for w in window_list
    ]

    if panel.empty:
        out = panel.copy()
        for col in out_cols:
            out[col] = pd.Series(dtype=float)
        return out

    result = panel.copy()
    for w, out_col in zip(window_list, out_cols):
        tmp_col = f"_size_mom_tmp_{w}"
        result = _add_size_mom_raw(result, window=w, col=tmp_col)
        result[out_col] = result[tmp_col]
        result = result.drop(columns=[tmp_col])

    return result


def add_value_momentum_interaction(
    panel: pd.DataFrame,
    *,
    mom_lookback: int = 252,
    mom_skip: int = 21,
) -> pd.DataFrame:
    """
    Add H-005 value-momentum interaction column ``val_mom_interact``.

    Features at date ``t`` use data through ``t``; labels are not added here.
    ``val_mom_interact = cs_rank(book_yield) * cs_rank(raw_momentum)``.
    Already in rank-product space; output is never CS-ranked again
    (``normalize`` is not offered).
    """
    _col = "val_mom_interact"
    required = {"date", "ticker", "close", "pb"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    return _add_val_mom_interact_raw(
        panel, mom_lookback=mom_lookback, mom_skip=mom_skip, col=_col,
    )


def add_value_momentum_distance(
    panel: pd.DataFrame,
    *,
    mom_lookback: int = 252,
    mom_skip: int = 21,
) -> pd.DataFrame:
    """
    Add H-005 value-momentum distance column ``val_mom_dist``.

    Features at date ``t`` use data through ``t``; labels are not added here.
    ``val_mom_dist = sqrt((1 - cs_rank(mom))^2 + (1 - cs_rank(book_yield))^2)``.

    The ideal point ``(1.0, 1.0)`` represents top-decile Value and top-decile
    Momentum. Already in rank-space; output is never CS-ranked again
    (``normalize`` is not offered).
    """
    _col = "val_mom_dist"
    required = {"date", "ticker", "close", "pb"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    return _add_val_mom_dist_raw(
        panel, mom_lookback=mom_lookback, mom_skip=mom_skip, col=_col,
    )


def add_value_momentum_residual(
    panel: pd.DataFrame,
    *,
    regression_window: int = 252,
    mom_lookback: int = 252,
    mom_skip: int = 21,
) -> pd.DataFrame:
    """
    Add H-005 value-momentum residual column ``val_mom_resid``.

    Features at date ``t`` use data through ``t``; labels are not added here.
    Standardised residual from rolling OLS of ``cs_rank(book_yield)`` on
    ``cs_rank(raw_momentum)``; NaN when ``std == 0``. Already a z-score;
    output is never CS-ranked (``normalize`` is not offered).
    """
    _col = "val_mom_resid"
    required = {"date", "ticker", "close", "pb"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    return _add_val_mom_resid_raw(
        panel,
        regression_window=regression_window,
        mom_lookback=mom_lookback,
        mom_skip=mom_skip,
        col=_col,
    )


def add_amihud(
    panel: pd.DataFrame,
    *,
    amihud_window: WindowSpec = 21,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add H-005 Amihud illiquidity column(s) ``amihud`` [``_{W}``].

    ``mean(|r| / (close * volume))`` over ``amihud_window``. Non-positive
    dollar volume → NaN that day. ``normalize=True`` (default) stores CS
    pct-rank within date.
    """
    window_list = normalize_windows(amihud_window)
    multi = len(window_list) > 1
    required = {"date", "ticker", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    out_cols = [
        regression_column_name("amihud", w, multi_window=multi) for w in window_list
    ]
    if panel.empty:
        out = panel.copy()
        for col in out_cols:
            out[col] = pd.Series(dtype=float)
        return out

    result = panel.copy()
    for w, out_col in zip(window_list, out_cols):
        tmp_col = f"_amihud_tmp_{w}"
        result = _add_amihud_raw(result, window=w, col=tmp_col)
        result[out_col] = result[tmp_col]
        if normalize:
            result[out_col] = cross_sectional_pct_rank(result, out_col)
        result = result.drop(columns=[tmp_col])
    return result


# ---------------------------------------------------------------------------
# H-010 · Short-selling pressure + supporting filing clock
# ---------------------------------------------------------------------------


def add_short_flow(
    panel: pd.DataFrame,
    *,
    smooth_window: WindowSpec = 5,
    baseline_window: WindowSpec = 60,
    mode: str = "abnormal",
) -> pd.DataFrame:
    """
    Add H-010 short-flow column(s) ``short_flow_{mode}`` [``_{S}_{B}``].

    Features use FINRA short volume through ``feature_date`` on the S1
    trade-date panel. Prefer ``add_short_flow_factors`` (fetches FINRA unless
    ``short_volume_data_exists=True``). Labels are not added here. Modes:

    - ``abnormal``: smoothed short-volume ratio z-scored vs own-history
      baseline (baseline excludes the current obs via ``shift(1)``).
    - ``ratio``: ``short_volume / total_volume`` (NaN when total ``<= 0``).
    - ``exempt_ratio``: ``short_exempt_volume / total_volume``.

    One ``(S, B)`` combo → ``short_flow_{mode}``; multiple combos →
    ``short_flow_{mode}_{S}_{B}``. Already unitless / own-history z-scored
    (abnormal) or a bounded ratio — there is no ``normalize`` kwarg.
    """
    if mode not in _VALID_SHORT_FLOW_MODES:
        raise ValueError(
            f"mode must be one of {sorted(_VALID_SHORT_FLOW_MODES)}, got {mode!r}"
        )

    smooth_list = normalize_windows(smooth_window)
    baseline_list = normalize_windows(baseline_window)
    combos = list(itertools.product(smooth_list, baseline_list))
    multi = len(combos) > 1
    stem = f"short_flow_{mode}"
    out_cols = [
        windowed_column_name(stem, s, b, multi=multi) for s, b in combos
    ]

    if mode == "exempt_ratio":
        required = {"date", "ticker", "short_exempt_volume", "total_volume"}
    else:
        required = {"date", "ticker", "short_volume", "total_volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    if panel.empty:
        out = panel.copy()
        for col in out_cols:
            out[col] = pd.Series(dtype=float)
        return out

    result = panel.copy()

    if mode == "ratio":
        ratio = short_volume_ratio(result["short_volume"], result["total_volume"])
        for out_col in out_cols:
            result[out_col] = ratio
        return result

    if mode == "exempt_ratio":
        ratio = short_volume_ratio(
            result["short_exempt_volume"], result["total_volume"]
        )
        for out_col in out_cols:
            result[out_col] = ratio
        return result

    # abnormal
    svr_tmp = "_short_volume_ratio_tmp"
    result = _add_short_volume_ratio_raw(result, col=svr_tmp)
    for (s, b), out_col in zip(combos, out_cols):
        abn_tmp = f"_abnormal_short_flow_tmp_{s}_{b}"
        result = _add_abnormal_short_flow_raw(
            result,
            smooth_window=s,
            baseline_window=b,
            col=abn_tmp,
            svr_col=svr_tmp,
        )
        result[out_col] = result[abn_tmp]
        result = result.drop(columns=[abn_tmp])
    return result.drop(columns=[svr_tmp])


def add_filing_event_clock(
    panel: pd.DataFrame,
    *,
    mode: str = "since",
) -> pd.DataFrame:
    """
    Add H-010 supporting filing-clock column ``filing_clock_{mode}``.

    Features use SEC filing anchors through ``feature_date`` on the S1
    trade-date panel. Prefer ``add_short_flow_factors`` (fetches filing clock
    unless ``filing_clock_data_exists=True``). Labels are not added here. Modes:

    - ``since``: calendar days since ``last_filed`` → ``filing_clock_since``.
    - ``expected_until``: signed calendar days until ``expected_next_filed``
      → ``filing_clock_expected_until`` (negative = overdue).

    Dense calendar-day counts — there is no ``normalize`` kwarg.
    """
    if mode not in _VALID_FILING_CLOCK_MODES:
        raise ValueError(
            f"mode must be one of {sorted(_VALID_FILING_CLOCK_MODES)}, got {mode!r}"
        )

    if mode == "since":
        _col = "filing_clock_since"
        required = {"date", "ticker", "last_filed"}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"panel missing columns: {sorted(missing)}")
        return _add_days_since_filing_raw(panel, col=_col)

    _col = "filing_clock_expected_until"
    required = {"date", "ticker", "expected_next_filed"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    return _add_expected_days_until_filing_raw(panel, col=_col)


# ---------------------------------------------------------------------------
# Family dispatchers (public API) — feature_subset selects which helpers run
# ---------------------------------------------------------------------------

from data.processing.feature_implementation.utilities import resolve_feature_subset

OBV_MOMENTUM_FEATURES: tuple[str, ...] = ("signed", "strict_zero", "soft")
GK_VOL_FEATURES: tuple[str, ...] = ("ratio", "log_ratio", "reversal")
IDIO_VOL_FEATURES: tuple[str, ...] = ("idio_vol",)
BETA_FEATURES: tuple[str, ...] = (
    "beta",
    "downside_beta",
    "upside_beta",
    "net_beta_spread",
    "rel_downside_beta",
    "rel_upside_beta",
    "blume_beta",
    "residual_mom",
    "smart_beta_smb",
    "smart_beta_hml",
    "smart_beta_mom",
    "smart_residual_mom",
    "market_corr",
    "r_squared",
    "rel_strength",
    "beta_mkt_interact",
    "mkt_ret",
    "mkt_vol",
    "mkt_near_52w",
)
SIZE_VALUE_FEATURES: tuple[str, ...] = (
    "book_yield",
    "earnings_yield",
    "log_mcap",
    "val_roc_pe",
    "val_roc_pb",
    "size_mom",
    "val_mom_interact",
    "val_mom_dist",
    "val_mom_resid",
    "amihud",
)
NEAR_52W_FEATURES: tuple[str, ...] = ("ratio", "log_drawdown")
MAX_LOTTERY_FEATURES: tuple[str, ...] = ("simple", "log")
GROSS_PROFITABILITY_FEATURES: tuple[str, ...] = ("gross_profitability",)
SHORT_FLOW_FEATURES: tuple[str, ...] = (
    "abnormal",
    "ratio",
    "exempt_ratio",
    "filing_since",
    "filing_expected_until",
)

_SHORT_VOLUME_FEATURE_IDS = frozenset({"abnormal", "ratio", "exempt_ratio"})
_FILING_CLOCK_FEATURE_IDS = frozenset({"filing_since", "filing_expected_until"})
_SHORT_VOLUME_RAW_COLS = ("short_volume", "short_exempt_volume", "total_volume")
_FILING_CLOCK_RAW_COLS = ("last_filed", "expected_next_filed")
_SIZE_VALUE_RAW_COLS = (
    "shares_outstanding",
    "book_equity",
    "eps_ttm",
    "market_cap",
    "pe",
    "pb",
)
_SIZE_VALUE_REQUIRED_COLS = ("market_cap", "pe", "pb")
_SIZE_VALUE_SEC_IDS = frozenset(SIZE_VALUE_FEATURES) - {"amihud"}
_GP_RAW_COLS = ("gross_profit_ttm", "assets", "gp_asset")
_GP_REQUIRED_COLS = ("gp_asset",)

_SPY_BETA_IDS = frozenset({
    "beta",
    "downside_beta",
    "upside_beta",
    "net_beta_spread",
    "rel_downside_beta",
    "rel_upside_beta",
    "blume_beta",
    "residual_mom",
    "market_corr",
    "r_squared",
    "rel_strength",
    "beta_mkt_interact",
    "mkt_ret",
    "mkt_vol",
})
_SMART_BETA_IDS = frozenset({
    "smart_beta_smb",
    "smart_beta_hml",
    "smart_beta_mom",
    "smart_residual_mom",
})
_MKT_OHLCV_IDS = frozenset({"mkt_near_52w"})


def _require_panel_base(panel: pd.DataFrame) -> None:
    missing = {"date", "ticker"} - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")


def _panel_info_col(panel: pd.DataFrame) -> str:
    return "feature_date" if "feature_date" in panel.columns else "date"


def _panel_tickers(panel: pd.DataFrame) -> list[str]:
    return sorted(
        {str(t).strip().upper() for t in panel["ticker"].tolist() if str(t).strip()}
    )


def _panel_date_bounds(
    panel: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    info = pd.to_datetime(panel[_panel_info_col(panel)])
    start = pd.Timestamp(start_date) if start_date is not None else info.min()
    end = pd.Timestamp(end_date) if end_date is not None else info.max()
    return start, end


def _price_panel_for_alt_fetch(panel: pd.DataFrame) -> pd.DataFrame:
    """OHLCV-like frame keyed by info date for SEC/FINRA ``price_panel``."""
    if "feature_date" in panel.columns and "close" in panel.columns:
        out = (
            panel[["feature_date", "ticker", "close"]]
            .rename(columns={"feature_date": "date"})
            .dropna(subset=["date"])
        )
        return out
    cols = [c for c in ("date", "ticker", "close") if c in panel.columns]
    return panel[cols].copy()


def _merge_alt_daily(
    panel: pd.DataFrame,
    alt: pd.DataFrame,
    value_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Left-merge alt daily columns onto ``panel``.

    S1 trade-date panels (with ``feature_date``): rename fetch ``date`` →
    ``feature_date`` and merge on ``[feature_date, ticker]``. Otherwise merge
    on ``[date, ticker]``. Existing overlapping ``value_cols`` are dropped
    before merge. Raw alt columns remain on the returned panel.
    """
    result = panel.copy()
    drop_cols = [c for c in value_cols if c in result.columns]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    if alt is None or alt.empty:
        for c in value_cols:
            result[c] = np.nan
        return result

    alt = alt.copy()
    alt["ticker"] = alt["ticker"].astype(str).str.upper()
    result["ticker"] = result["ticker"].astype(str).str.upper()

    if "feature_date" in result.columns:
        if "date" in alt.columns:
            alt = alt.rename(columns={"date": "feature_date"})
        alt["feature_date"] = pd.to_datetime(alt["feature_date"])
        result["feature_date"] = pd.to_datetime(result["feature_date"])
        merge_keys = ["feature_date", "ticker"]
    else:
        alt["date"] = pd.to_datetime(alt["date"])
        result["date"] = pd.to_datetime(result["date"])
        merge_keys = ["date", "ticker"]

    merge_cols = merge_keys + [c for c in value_cols if c in alt.columns]
    for c in value_cols:
        if c not in alt.columns:
            alt[c] = np.nan
            merge_cols.append(c)
    return result.merge(alt[merge_cols], on=merge_keys, how="left")


def _attach_short_volume_daily(
    panel: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_dir: str | None = None,
    facilities: tuple[str, ...] | None = None,
    price_panel: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Fetch FINRA short volume and left-merge onto ``panel``."""
    from data.ingestion.alternative_data.finra_short_volume import (
        DEFAULT_CACHE_DIR,
        DEFAULT_FACILITIES,
        DEFAULT_MAX_WORKERS,
        fetch_short_volume_daily,
    )

    _require_panel_base(panel)
    start, end = _panel_date_bounds(panel, start_date=start_date, end_date=end_date)
    kwargs: dict = {
        "start_date": start,
        "end_date": end,
        "cache_dir": cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR,
        "facilities": facilities if facilities is not None else DEFAULT_FACILITIES,
        "price_panel": (
            price_panel if price_panel is not None else _price_panel_for_alt_fetch(panel)
        ),
        "max_workers": (
            max_workers if max_workers is not None else DEFAULT_MAX_WORKERS
        ),
    }
    alt = fetch_short_volume_daily(_panel_tickers(panel), **kwargs)
    return _merge_alt_daily(panel, alt, _SHORT_VOLUME_RAW_COLS)


def _attach_filing_clock_daily(
    panel: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_dir: str | None = None,
    forms: tuple[str, ...] | None = None,
    price_panel: pd.DataFrame | None = None,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """Fetch SEC filing-clock anchors and left-merge onto ``panel``."""
    from data.ingestion.alternative_data.sec_companyfacts import (
        DEFAULT_CACHE_DIR,
        fetch_filing_clock_daily,
    )

    _require_panel_base(panel)
    start, end = _panel_date_bounds(panel, start_date=start_date, end_date=end_date)
    kwargs: dict = {
        "start_date": start,
        "end_date": end,
        "cache_dir": cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR,
        "price_panel": (
            price_panel if price_panel is not None else _price_panel_for_alt_fetch(panel)
        ),
        "user_agent": user_agent,
    }
    if forms is not None:
        kwargs["forms"] = forms
    alt = fetch_filing_clock_daily(_panel_tickers(panel), **kwargs)
    return _merge_alt_daily(panel, alt, _FILING_CLOCK_RAW_COLS)


def _attach_size_value_daily(
    panel: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_dir: str | None = None,
    price_panel: pd.DataFrame | None = None,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """Fetch SEC size/value fields and left-merge onto ``panel``."""
    from data.ingestion.alternative_data.sec_companyfacts import (
        DEFAULT_CACHE_DIR,
        fetch_size_value_daily,
    )

    _require_panel_base(panel)
    start, end = _panel_date_bounds(panel, start_date=start_date, end_date=end_date)
    alt = fetch_size_value_daily(
        _panel_tickers(panel),
        start_date=start,
        end_date=end,
        cache_dir=cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR,
        price_panel=(
            price_panel if price_panel is not None else _price_panel_for_alt_fetch(panel)
        ),
        user_agent=user_agent,
    )
    return _merge_alt_daily(panel, alt, _SIZE_VALUE_RAW_COLS)


def _attach_gross_profitability_daily(
    panel: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_dir: str | None = None,
    price_panel: pd.DataFrame | None = None,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """Fetch SEC gross-profitability fields and left-merge onto ``panel``."""
    from data.ingestion.alternative_data.sec_companyfacts import (
        DEFAULT_CACHE_DIR,
        fetch_gross_profitability_daily,
    )

    _require_panel_base(panel)
    start, end = _panel_date_bounds(panel, start_date=start_date, end_date=end_date)
    alt = fetch_gross_profitability_daily(
        _panel_tickers(panel),
        start_date=start,
        end_date=end,
        cache_dir=cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR,
        price_panel=(
            price_panel if price_panel is not None else _price_panel_for_alt_fetch(panel)
        ),
        user_agent=user_agent,
    )
    return _merge_alt_daily(panel, alt, _GP_RAW_COLS)


def add_obv_momentum_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    lookback: WindowSpec = 252,
    skip: WindowSpec = 21,
    obv_window: WindowSpec = 20,
) -> pd.DataFrame:
    """Add H-001 OBV-confirmed momentum columns for each id in ``feature_subset``."""
    ids = resolve_feature_subset(
        feature_subset, OBV_MOMENTUM_FEATURES, name="add_obv_momentum_factors"
    )
    result = panel
    for mode in ids:
        result = add_obv_confirmed_momentum(
            result,
            lookback=lookback,
            skip=skip,
            obv_window=obv_window,
            mode=mode,
        )
    return result


def add_gk_vol_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    gk_window: WindowSpec = 5,
    realised_window: WindowSpec = 20,
    normalize: bool = True,
) -> pd.DataFrame:
    """Add H-002 GK / realised vol columns for each id in ``feature_subset``."""
    ids = resolve_feature_subset(
        feature_subset, GK_VOL_FEATURES, name="add_gk_vol_factors"
    )
    result = panel
    for mode in ids:
        result = add_gk_vol_ratio(
            result,
            gk_window=gk_window,
            realised_window=realised_window,
            mode=mode,
            normalize=normalize,
        )
    return result


def add_idio_vol_factors(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    windows: WindowSpec = 20,
    normalize: bool = True,
    market_col: str = "market_log_ret",
) -> pd.DataFrame:
    """Add H-003 idiosyncratic-vol columns (``feature_subset`` must be ``idio_vol``)."""
    ids = resolve_feature_subset(
        feature_subset, IDIO_VOL_FEATURES, name="add_idio_vol_factors"
    )
    result = panel
    if "idio_vol" in ids:
        result = add_idiosyncratic_vol(
            result,
            market_returns,
            windows=windows,
            normalize=normalize,
            market_col=market_col,
        )
    return result


def add_beta_factors(
    panel: pd.DataFrame,
    market_returns: pd.DataFrame | None = None,
    ff_factors: pd.DataFrame | None = None,
    *,
    feature_subset: Sequence[str] | None = None,
    windows: WindowSpec = 252,
    normalize: bool = True,
    market_col: str = "market_log_ret",
    benchmark: str = "spy",
    formation_window: WindowSpec = 252,
    skip: WindowSpec = 21,
    mkt_horizon: WindowSpec = 5,
    mkt_ret_windows: WindowSpec = 5,
    mkt_vol_windows: WindowSpec = 21,
    mkt_near_windows: WindowSpec = 252,
    market_ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add H-004 beta-family columns selected by ``feature_subset``."""
    ids = resolve_feature_subset(
        feature_subset, BETA_FEATURES, name="add_beta_factors"
    )
    need_mkt = bool(set(ids) & _SPY_BETA_IDS)
    need_ff = bool(set(ids) & _SMART_BETA_IDS)
    need_ohlcv = bool(set(ids) & _MKT_OHLCV_IDS)
    if need_mkt and market_returns is None:
        raise ValueError(
            "add_beta_factors: market_returns is required for "
            f"{sorted(set(ids) & _SPY_BETA_IDS)}"
        )
    if need_ff and ff_factors is None:
        raise ValueError(
            "add_beta_factors: ff_factors is required for "
            f"{sorted(set(ids) & _SMART_BETA_IDS)}"
        )
    if need_ohlcv and market_ohlcv is None:
        raise ValueError(
            "add_beta_factors: market_ohlcv is required for "
            f"{sorted(set(ids) & _MKT_OHLCV_IDS)}"
        )

    result = panel
    id_set = set(ids)

    if "beta" in id_set:
        result = add_beta(
            result,
            market_returns,
            benchmark=benchmark,
            windows=windows,
            normalize=normalize,
            market_col=market_col,
        )
    if "downside_beta" in id_set:
        result = add_downside_beta(
            result,
            market_returns,
            windows=windows,
            normalize=normalize,
            market_col=market_col,
        )
    if "upside_beta" in id_set:
        result = add_upside_beta(
            result,
            market_returns,
            windows=windows,
            normalize=normalize,
            market_col=market_col,
        )
    if "net_beta_spread" in id_set:
        result = add_net_beta_spread(
            result,
            market_returns,
            windows=windows,
            normalize=normalize,
            market_col=market_col,
        )
    if "rel_downside_beta" in id_set:
        result = add_relative_downside_beta(
            result,
            market_returns,
            windows=windows,
            normalize=normalize,
            market_col=market_col,
        )
    if "rel_upside_beta" in id_set:
        result = add_relative_upside_beta(
            result,
            market_returns,
            windows=windows,
            normalize=normalize,
            market_col=market_col,
        )
    if "blume_beta" in id_set:
        result = add_blume_beta(
            result,
            market_returns,
            windows=windows,
            market_col=market_col,
        )
    if "residual_mom" in id_set:
        result = add_residual_momentum(
            result,
            market_returns,
            benchmark=benchmark,
            formation_window=formation_window,
            skip=skip,
            market_col=market_col,
        )

    smart_requested = [s for s in ("smart_beta_smb", "smart_beta_hml", "smart_beta_mom") if s in id_set]
    if smart_requested:
        before_cols = set(result.columns)
        result = add_beta(
            result,
            ff_factors,
            benchmark="ff",
            windows=windows,
            normalize=normalize,
            market_col=market_col,
        )
        # Drop smart columns that were not requested (helper writes all three).
        window_list = normalize_windows(windows)
        multi = len(window_list) > 1
        keep_stems = set(smart_requested)
        for stem in ("smart_beta_smb", "smart_beta_hml", "smart_beta_mom"):
            if stem in keep_stems:
                continue
            for w in window_list:
                col = regression_column_name(stem, w, multi_window=multi)
                if col in result.columns and col not in before_cols:
                    result = result.drop(columns=[col])

    if "smart_residual_mom" in id_set:
        result = add_residual_momentum(
            result,
            ff_factors,
            benchmark="ff",
            formation_window=formation_window,
            skip=skip,
            market_col=market_col,
        )

    if "market_corr" in id_set:
        result = add_market_corr(
            result, market_returns, windows=windows, market_col=market_col,
        )
    if "r_squared" in id_set:
        result = add_r_squared(
            result, market_returns, windows=windows, market_col=market_col,
        )
    if "rel_strength" in id_set:
        result = add_rel_strength(
            result, market_returns, windows=windows, market_col=market_col,
        )
    if "beta_mkt_interact" in id_set:
        result = add_beta_mkt_interact(
            result,
            market_returns,
            windows=windows,
            mkt_horizon=mkt_horizon,
            market_col=market_col,
        )
    if "mkt_ret" in id_set:
        result = add_mkt_ret(
            result,
            market_returns,
            windows=mkt_ret_windows,
            market_col=market_col,
        )
    if "mkt_vol" in id_set:
        result = add_mkt_vol(
            result,
            market_returns,
            windows=mkt_vol_windows,
            market_col=market_col,
        )
    if "mkt_near_52w" in id_set:
        result = add_mkt_near_52w(
            result, market_ohlcv, windows=mkt_near_windows,
        )
    return result


def add_size_value_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    size_value_data_exists: bool = False,
    normalize: bool = True,
    window: WindowSpec = 63,
    mom_lookback: int = 252,
    mom_skip: int = 21,
    regression_window: int = 252,
    amihud_window: WindowSpec = 21,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_dir: str | None = None,
    price_panel: pd.DataFrame | None = None,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """
    Add H-005 size/value columns selected by ``feature_subset``.

    ``size_value_data_exists``:
        Applies when the subset needs SEC size/value inputs (anything other
        than amihud-only). If False (default), call
        ``fetch_size_value_daily`` and merge raw columns onto the panel
        (S1: on ``feature_date`` when present; else on ``date``). If True,
        require ``market_cap`` / ``pe`` / ``pb`` already on the panel.
        Amihud-only requests skip the SEC fetch entirely.

    Raw SEC columns (``shares_outstanding``, ``book_equity``, ``eps_ttm``,
    ``market_cap``, ``pe``, ``pb``) remain on the returned panel so later
    calls can pass ``size_value_data_exists=True``.
    """
    ids = resolve_feature_subset(
        feature_subset, SIZE_VALUE_FEATURES, name="add_size_value_factors"
    )
    id_set = set(ids)
    need_sec = bool(id_set & _SIZE_VALUE_SEC_IDS)

    if need_sec:
        if size_value_data_exists:
            missing = set(_SIZE_VALUE_REQUIRED_COLS) - set(panel.columns)
            if missing:
                raise ValueError(f"panel missing columns: {sorted(missing)}")
            result = panel.copy()
        else:
            result = _attach_size_value_daily(
                panel,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                price_panel=price_panel,
                user_agent=user_agent,
            )
    else:
        result = panel

    if "book_yield" in id_set:
        result = add_book_yield(result, normalize=normalize)
    if "earnings_yield" in id_set:
        result = add_earnings_yield(result, normalize=normalize)
    if "log_mcap" in id_set:
        result = add_log_mcap(result, normalize=normalize)
    if "val_roc_pe" in id_set:
        result = add_valuation_roc(result, metric="pe", window=window)
    if "val_roc_pb" in id_set:
        result = add_valuation_roc(result, metric="pb", window=window)
    if "size_mom" in id_set:
        result = add_size_momentum(result, window=window)
    if "val_mom_interact" in id_set:
        result = add_value_momentum_interaction(
            result, mom_lookback=mom_lookback, mom_skip=mom_skip
        )
    if "val_mom_dist" in id_set:
        result = add_value_momentum_distance(
            result, mom_lookback=mom_lookback, mom_skip=mom_skip
        )
    if "val_mom_resid" in id_set:
        result = add_value_momentum_residual(
            result,
            regression_window=regression_window,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
        )
    if "amihud" in id_set:
        result = add_amihud(
            result, amihud_window=amihud_window, normalize=normalize,
        )
    return result


def add_near_52w_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    window: WindowSpec = 252,
    normalize: bool = False,
) -> pd.DataFrame:
    """Add H-006 near-52w columns for each id in ``feature_subset``."""
    ids = resolve_feature_subset(
        feature_subset, NEAR_52W_FEATURES, name="add_near_52w_factors"
    )
    result = panel
    for mode in ids:
        result = add_near_52w(
            result, window=window, mode=mode, normalize=normalize
        )
    return result


def add_max_lottery_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    n_extreme: WindowSpec = 5,
    window: WindowSpec = 21,
    normalize: bool = True,
    add_residuals: bool = False,
    idio_vol_col: str = "idio_vol",
) -> pd.DataFrame:
    """Add H-007 MAX lottery columns for each id in ``feature_subset``."""
    ids = resolve_feature_subset(
        feature_subset, MAX_LOTTERY_FEATURES, name="add_max_lottery_factors"
    )
    result = panel
    for mode in ids:
        result = add_max_lottery(
            result,
            n_extreme=n_extreme,
            window=window,
            mode=mode,
            normalize=normalize,
            add_residuals=add_residuals,
            idio_vol_col=idio_vol_col,
        )
    return result


def add_gross_profitability_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    gross_profitability_data_exists: bool = False,
    normalize: bool = True,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_dir: str | None = None,
    price_panel: pd.DataFrame | None = None,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """
    Add H-008 gross-profitability column when selected in ``feature_subset``.

    ``gross_profitability_data_exists``:
        If False (default), call ``fetch_gross_profitability_daily`` and merge
        ``gross_profit_ttm`` / ``assets`` / ``gp_asset`` onto the panel
        (S1: on ``feature_date`` when present; else on ``date``). If True,
        require ``gp_asset`` already on the panel.

    Raw SEC columns remain on the returned panel so later calls can pass
    ``gross_profitability_data_exists=True``.
    """
    ids = resolve_feature_subset(
        feature_subset,
        GROSS_PROFITABILITY_FEATURES,
        name="add_gross_profitability_factors",
    )
    result = panel
    if "gross_profitability" in ids:
        if gross_profitability_data_exists:
            missing = set(_GP_REQUIRED_COLS) - set(panel.columns)
            if missing:
                raise ValueError(f"panel missing columns: {sorted(missing)}")
            result = panel.copy()
        else:
            result = _attach_gross_profitability_daily(
                panel,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                price_panel=price_panel,
                user_agent=user_agent,
            )
        result = add_gross_profitability(result, normalize=normalize)
    return result


def add_short_flow_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    short_volume_data_exists: bool = False,
    filing_clock_data_exists: bool = False,
    smooth_window: WindowSpec = 5,
    baseline_window: WindowSpec = 60,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_dir: str | None = None,
    price_panel: pd.DataFrame | None = None,
    facilities: tuple[str, ...] | None = None,
    forms: tuple[str, ...] | None = None,
    user_agent: str | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """
    Add H-010 short-flow / filing-clock columns selected by ``feature_subset``.

    ``short_volume_data_exists``:
        Applies when the subset needs FINRA short volume
        (``abnormal`` / ``ratio`` / ``exempt_ratio``). If False (default),
        call ``fetch_short_volume_daily`` and merge raw volume columns onto
        the panel. If True, require those columns already present.

    ``filing_clock_data_exists``:
        Applies when the subset needs filing anchors
        (``filing_since`` / ``filing_expected_until``). If False (default),
        call ``fetch_filing_clock_daily`` and merge ``last_filed`` /
        ``expected_next_filed``. If True, require those columns already
        present.

    Fetches only what the requested subset needs. Raw alt columns remain on
    the returned panel so later calls can pass the matching exists flag.
    S1 panels merge on ``feature_date`` when present; else on ``date``.
    """
    ids = resolve_feature_subset(
        feature_subset, SHORT_FLOW_FEATURES, name="add_short_flow_factors"
    )
    id_set = set(ids)
    need_short = bool(id_set & _SHORT_VOLUME_FEATURE_IDS)
    need_filing = bool(id_set & _FILING_CLOCK_FEATURE_IDS)

    result = panel
    if need_short:
        if short_volume_data_exists:
            missing = set(_SHORT_VOLUME_RAW_COLS) - set(result.columns)
            if missing:
                raise ValueError(f"panel missing columns: {sorted(missing)}")
            result = result.copy()
        else:
            result = _attach_short_volume_daily(
                result,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                facilities=facilities,
                price_panel=price_panel,
                max_workers=max_workers,
            )

    if need_filing:
        if filing_clock_data_exists:
            missing = set(_FILING_CLOCK_RAW_COLS) - set(result.columns)
            if missing:
                raise ValueError(f"panel missing columns: {sorted(missing)}")
            if result is panel:
                result = result.copy()
        else:
            result = _attach_filing_clock_daily(
                result,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                forms=forms,
                price_panel=price_panel,
                user_agent=user_agent,
            )

    for mode in ("abnormal", "ratio", "exempt_ratio"):
        if mode in id_set:
            result = add_short_flow(
                result,
                smooth_window=smooth_window,
                baseline_window=baseline_window,
                mode=mode,
            )
    if "filing_since" in id_set:
        result = add_filing_event_clock(result, mode="since")
    if "filing_expected_until" in id_set:
        result = add_filing_event_clock(result, mode="expected_until")
    return result


GDELT_SENTIMENT_FEATURES: tuple[str, ...] = (
    "tone",
    "attention",
    "abnormal_tone",
    "abnormal_attention",
    "tone_x_attention",
    "tone_mom",
)
VOLUME_FEATURES: tuple[str, ...] = ("abnormal_volume",)
OPEN_REALIZED_VOL_FEATURES: tuple[str, ...] = ("open_realized_vol",)
ATR_FEATURES: tuple[str, ...] = ("atr",)
TALIB_FEATURES: tuple[str, ...] = ("rsi", "adx", "mfi", "bb_percent_b")


def _attach_gdelt_sentiment_daily(
    panel: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_dir: str | None = None,
    company_name_map_path: str | None = None,
    use_bigquery: bool = True,
    live_n_files: int = 0,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Fetch GDELT daily tone/attention and left-merge onto ``panel``.

    S1 trade-date panels (with ``feature_date``): merge on
    ``[feature_date, ticker]`` after renaming fetch ``date`` → ``feature_date``.
    Otherwise merge on ``[date, ticker]``.
    """
    from data.ingestion.alternative_data.sentiment.gdelt_fetcher import (
        DEFAULT_CACHE_DIR,
        fetch_gdelt_sentiment_daily,
    )

    base = {"date", "ticker"}
    missing = base - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    info_col = "feature_date" if "feature_date" in panel.columns else "date"
    info = pd.to_datetime(panel[info_col])
    start = start_date if start_date is not None else info.min()
    end = end_date if end_date is not None else info.max()

    tickers = sorted(
        {str(t).strip().upper() for t in panel["ticker"].tolist() if str(t).strip()}
    )
    gd = fetch_gdelt_sentiment_daily(
        tickers,
        start_date=start,
        end_date=end,
        cache_dir=cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR,
        company_name_map_path=company_name_map_path,
        use_bigquery=use_bigquery,
        live_n_files=live_n_files,
        resume=resume,
    )

    result = panel.copy()
    drop_cols = [c for c in ("median_tone", "n_articles") if c in result.columns]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    if gd.empty:
        result["median_tone"] = np.nan
        result["n_articles"] = np.nan
        return result

    gd = gd.copy()
    gd["ticker"] = gd["ticker"].astype(str).str.upper()
    result["ticker"] = result["ticker"].astype(str).str.upper()

    if "feature_date" in result.columns:
        gd = gd.rename(columns={"date": "feature_date"})
        gd["feature_date"] = pd.to_datetime(gd["feature_date"])
        result["feature_date"] = pd.to_datetime(result["feature_date"])
        merge_keys = ["feature_date", "ticker"]
    else:
        gd["date"] = pd.to_datetime(gd["date"])
        result["date"] = pd.to_datetime(result["date"])
        merge_keys = ["date", "ticker"]

    merge_cols = merge_keys + ["median_tone", "n_articles"]
    return result.merge(gd[merge_cols], on=merge_keys, how="left")


def add_gdelt_sentiment_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    sentiment_data_exists: bool = False,
    window: WindowSpec = 5,
    smooth_window: WindowSpec = 5,
    baseline_window: WindowSpec = 60,
    short_window: WindowSpec = 5,
    long_window: WindowSpec = 21,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_dir: str | None = None,
    company_name_map_path: str | None = None,
    use_bigquery: bool = True,
    live_n_files: int = 0,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Add H-009 GDELT sentiment column(s) selected by ``feature_subset``.

    ``sentiment_data_exists``:
        If False (default), call ``fetch_gdelt_sentiment_daily`` and merge
        ``median_tone`` / ``n_articles`` onto the panel (S1: on
        ``feature_date`` when present; else on ``date``). If True, require
        those columns already on the panel (research / pre-merged path).

    Fetch passthrough kwargs (``start_date``, ``end_date``, ``cache_dir``,
    ``company_name_map_path``, ``use_bigquery``, ``live_n_files``, ``resume``)
    apply only when ``sentiment_data_exists`` is False.

    No store-side ``normalize`` — raw / own-history z only.

    IDs: ``tone``, ``attention``, ``abnormal_tone``, ``abnormal_attention``,
    ``tone_x_attention``, ``tone_mom``. Columns are ``gdelt_{id}`` with window
    suffixes when multiple combos are requested.
    """
    from data.processing.feature_implementation.gdelt_sentiment import (
        add_gdelt_abnormal_attention,
        add_gdelt_abnormal_tone,
        add_gdelt_attention,
        add_gdelt_tone,
        add_gdelt_tone_mom,
        add_gdelt_tone_x_attention,
    )

    ids = resolve_feature_subset(
        feature_subset, GDELT_SENTIMENT_FEATURES, name="add_gdelt_sentiment_factors"
    )

    if sentiment_data_exists:
        required = {"date", "ticker", "median_tone", "n_articles"}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"panel missing columns: {sorted(missing)}")
        result = panel.copy()
    else:
        result = _attach_gdelt_sentiment_daily(
            panel,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            company_name_map_path=company_name_map_path,
            use_bigquery=use_bigquery,
            live_n_files=live_n_files,
            resume=resume,
        )

    windows = normalize_windows(window)
    smooths = normalize_windows(smooth_window)
    baselines = normalize_windows(baseline_window)
    shorts = normalize_windows(short_window)
    longs = normalize_windows(long_window)

    id_set = set(ids)

    if "tone" in id_set:
        multi = len(windows) > 1
        for w in windows:
            tmp = f"_gdelt_tone_tmp_{w}"
            result = add_gdelt_tone(result, window=w, col=tmp)
            out = windowed_column_name("gdelt_tone", w, multi=multi)
            result[out] = result[tmp]
            result = result.drop(columns=[tmp])

    if "attention" in id_set:
        multi = len(windows) > 1
        for w in windows:
            tmp = f"_gdelt_att_tmp_{w}"
            result = add_gdelt_attention(result, window=w, col=tmp)
            out = windowed_column_name("gdelt_attention", w, multi=multi)
            result[out] = result[tmp]
            result = result.drop(columns=[tmp])

    if "abnormal_tone" in id_set:
        combos = list(itertools.product(smooths, baselines))
        multi = len(combos) > 1
        for s, b in combos:
            tmp = f"_gdelt_abn_tone_tmp_{s}_{b}"
            result = add_gdelt_abnormal_tone(
                result, smooth_window=s, baseline_window=b, col=tmp
            )
            out = windowed_column_name("gdelt_abnormal_tone", s, b, multi=multi)
            result[out] = result[tmp]
            result = result.drop(columns=[tmp])

    if "abnormal_attention" in id_set:
        combos = list(itertools.product(smooths, baselines))
        multi = len(combos) > 1
        for s, b in combos:
            tmp = f"_gdelt_abn_att_tmp_{s}_{b}"
            result = add_gdelt_abnormal_attention(
                result, smooth_window=s, baseline_window=b, col=tmp
            )
            out = windowed_column_name(
                "gdelt_abnormal_attention", s, b, multi=multi
            )
            result[out] = result[tmp]
            result = result.drop(columns=[tmp])

    if "tone_x_attention" in id_set:
        multi = len(windows) > 1
        for w in windows:
            tmp = f"_gdelt_txa_tmp_{w}"
            result = add_gdelt_tone_x_attention(result, window=w, col=tmp)
            out = windowed_column_name("gdelt_tone_x_attention", w, multi=multi)
            result[out] = result[tmp]
            result = result.drop(columns=[tmp])

    if "tone_mom" in id_set:
        combos = list(itertools.product(shorts, longs))
        for sh, lo in combos:
            if lo <= sh:
                raise ValueError(
                    f"long_window must be > short_window for every combo; "
                    f"got short={sh}, long={lo}"
                )
        multi = len(combos) > 1
        for sh, lo in combos:
            tmp = f"_gdelt_tmom_tmp_{sh}_{lo}"
            result = add_gdelt_tone_mom(
                result, short_window=sh, long_window=lo, col=tmp
            )
            out = windowed_column_name("gdelt_tone_mom", sh, lo, multi=multi)
            result[out] = result[tmp]
            result = result.drop(columns=[tmp])

    return result


def add_volume_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    smooth_window: WindowSpec = 5,
    baseline_window: WindowSpec = 60,
) -> pd.DataFrame:
    """
    Add abnormal-volume column(s) selected by ``feature_subset``.

    Features at ``t`` use volume through close of ``t`` (S1: through
    ``feature_date`` bars on the trade row). No store-side ``normalize`` —
    output is already an own-history z-score. ID: ``abnormal_volume``.
    """
    from data.processing.feature_implementation.volume_features import (
        add_abnormal_volume_multi,
    )

    ids = resolve_feature_subset(
        feature_subset, VOLUME_FEATURES, name="add_volume_factors"
    )
    result = panel
    if "abnormal_volume" in ids:
        result = add_abnormal_volume_multi(
            result,
            smooth_window=smooth_window,
            baseline_window=baseline_window,
        )
    return result


def add_open_realized_vol_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    windows: WindowSpec = 21,
    open_col: str = "open",
    pit_shift: int = 1,
) -> pd.DataFrame:
    """
    Add open-to-open trailing realized-vol column(s) selected by ``feature_subset``.

    ID: ``open_realized_vol``. Default window 21 with ``pit_shift=1`` so the
    value on date ``d`` uses only open-to-open returns before ``d`` (S1
    pre-open). Math lives in ``feature_implementation.realized_vol``.
    """
    from data.processing.feature_implementation.realized_vol import (
        add_open_realized_vol_multi,
    )

    ids = resolve_feature_subset(
        feature_subset,
        OPEN_REALIZED_VOL_FEATURES,
        name="add_open_realized_vol_factors",
    )
    result = panel
    if "open_realized_vol" in ids:
        result = add_open_realized_vol_multi(
            result,
            windows=windows,
            open_col=open_col,
            pit_shift=pit_shift,
        )
    return result


def add_atr_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    windows: WindowSpec = 14,
    pit_shift: int = 1,
) -> pd.DataFrame:
    """
    Add Wilder ATR column(s) selected by ``feature_subset``.

    ID: ``atr``. Default window 14 with ``pit_shift=1`` (S1 pre-open).
    Math lives in ``feature_implementation.atr`` (Wilder smoother, not SMA-TR).
    """
    from data.processing.feature_implementation.atr import add_wilder_atr_multi

    ids = resolve_feature_subset(
        feature_subset, ATR_FEATURES, name="add_atr_factors"
    )
    result = panel
    if "atr" in ids:
        result = add_wilder_atr_multi(
            result, windows=windows, pit_shift=pit_shift
        )
    return result


def add_talib_factors(
    panel: pd.DataFrame,
    *,
    feature_subset: Sequence[str] | None = None,
    timeperiod: WindowSpec = 14,
    bb_timeperiod: WindowSpec | None = None,
    bb_nbdev: float = 2.0,
) -> pd.DataFrame:
    """
    Add TA-Lib technical columns selected by ``feature_subset``.

    IDs: ``rsi``, ``adx``, ``mfi``, ``bb_percent_b``. No store-side
    ``normalize``. Default ``timeperiod=14``; Bollinger uses
    ``bb_timeperiod`` (default 20 when None) and fixed ``bb_nbdev=2.0``.
    """
    from data.processing.feature_implementation.talib_features import (
        add_adx_multi,
        add_bb_percent_b_multi,
        add_mfi_multi,
        add_rsi_multi,
    )

    ids = resolve_feature_subset(
        feature_subset, TALIB_FEATURES, name="add_talib_factors"
    )
    bb_periods: WindowSpec = 20 if bb_timeperiod is None else bb_timeperiod

    result = panel
    id_set = set(ids)
    if "rsi" in id_set:
        result = add_rsi_multi(result, timeperiod=timeperiod)
    if "adx" in id_set:
        result = add_adx_multi(result, timeperiod=timeperiod)
    if "mfi" in id_set:
        result = add_mfi_multi(result, timeperiod=timeperiod)
    if "bb_percent_b" in id_set:
        result = add_bb_percent_b_multi(
            result, timeperiod=bb_periods, nbdev=bb_nbdev
        )
    return result
