"""S2 cointegration store: public dispatchers over cointegration math (no formulae here)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from data.processing.feature_implementation.cointegration import (
    COINT_PVALUE,
    CointResult,
    adaptive_zscore,
    ewm_zscore,
    kalman_hedge,
    ou_half_life,
    ou_residual_score,
    residual_variance_ratio,
    rolling_adf_pvalue,
    rolling_hedge,
    rolling_ou_half_life,
    rolling_zscore,
    test_cointegration,
    to_log_price,
)
from data.processing.feature_implementation.utilities import resolve_feature_subset

COINT_METRICS: tuple[str, ...] = ("adf_pvalue", "variance_jump")

_OHLC_COLS: tuple[str, ...] = ("open", "high", "low", "close")

_PANEL_BASE_COLS: tuple[str, ...] = (
    "date",
    "pair_id",
    "ticker_y",
    "ticker_x",
    "open_y",
    "high_y",
    "low_y",
    "close_y",
    "open_x",
    "high_x",
    "low_x",
    "close_x",
    "alpha",
    "beta",
    "spread",
    "z",
    "half_life",
)


def _validate_pair_inputs(y: pd.Series, x: pd.Series) -> None:
    """Require identical monotonic DatetimeIndex with no duplicates."""
    for name, s in (("y", y), ("x", x)):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex, got {type(s.index).__name__}")
        if not s.index.is_monotonic_increasing:
            raise ValueError(f"{name} DatetimeIndex must be monotonic increasing")
        if s.index.has_duplicates:
            raise ValueError(f"{name} DatetimeIndex must not contain duplicates")
    if not y.index.equals(x.index):
        raise ValueError("y and x must share an identical DatetimeIndex (no silent alignment)")


def _as_price_series(prices_by_ticker: Mapping[str, pd.Series], ticker: str) -> pd.Series:
    if ticker not in prices_by_ticker:
        raise KeyError(f"missing price series for ticker {ticker!r}")
    s = prices_by_ticker[ticker]
    if not isinstance(s, pd.Series):
        raise TypeError(f"prices for {ticker!r} must be a Series, got {type(s).__name__}")
    out = s.astype(float).copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"price series for {ticker!r} has duplicate dates")
    return out


def _align_pair_closes(
    prices_by_ticker: Mapping[str, pd.Series],
    ticker_a: str,
    ticker_b: str,
) -> tuple[pd.Series, pd.Series]:
    """Inner-join closes on the mutual date index (pair starts at first shared bar)."""
    a = _as_price_series(prices_by_ticker, ticker_a)
    b = _as_price_series(prices_by_ticker, ticker_b)
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return (
            pd.Series(dtype=float, index=pd.DatetimeIndex([])),
            pd.Series(dtype=float, index=pd.DatetimeIndex([])),
        )
    return a.loc[idx], b.loc[idx]


def _as_ohlc_frame(ohlc_by_ticker: Mapping[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
    if ticker not in ohlc_by_ticker:
        raise KeyError(f"missing OHLC frame for ticker {ticker!r}")
    df = ohlc_by_ticker[ticker]
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"OHLC for {ticker!r} must be a DataFrame, got {type(df).__name__}")
    missing = [c for c in _OHLC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC for {ticker!r} missing columns: {missing}")
    out = df.loc[:, list(_OHLC_COLS)].astype(float).copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"OHLC for {ticker!r} has duplicate dates")
    if not out.index.is_monotonic_increasing:
        raise ValueError(f"OHLC for {ticker!r} DatetimeIndex must be monotonic increasing")
    return out


def _align_pair_ohlc(
    ohlc_by_ticker: Mapping[str, pd.DataFrame],
    ticker_a: str,
    ticker_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join OHLC frames on mutual dates where both closes are finite."""
    a = _as_ohlc_frame(ohlc_by_ticker, ticker_a)
    b = _as_ohlc_frame(ohlc_by_ticker, ticker_b)
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        empty = pd.DataFrame(columns=list(_OHLC_COLS), index=pd.DatetimeIndex([]))
        return empty, empty
    a = a.loc[idx]
    b = b.loc[idx]
    ok = a["close"].notna() & b["close"].notna()
    a = a.loc[ok]
    b = b.loc[ok]
    return a, b


def compute_static_hedge_spread(
    y_price: pd.Series,
    x_price: pd.Series,
    *,
    window: int = 252,
) -> pd.DataFrame:
    """Rolling OLS on log prices; spread ``s_t = y_t - alpha_t - beta_t * x_t``."""
    _validate_pair_inputs(y_price, x_price)
    return rolling_hedge(to_log_price(y_price), to_log_price(x_price), window=window)


def compute_kalman_hedge_spread(
    y_price: pd.Series,
    x_price: pd.Series,
    *,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
    burn_in: int = 30,
) -> pd.DataFrame:
    """Kalman hedge on log prices; spread from prior state (see spread_var / z_innov in math)."""
    _validate_pair_inputs(y_price, x_price)
    return kalman_hedge(
        to_log_price(y_price),
        to_log_price(x_price),
        delta=delta,
        obs_var=obs_var,
        burn_in=burn_in,
    )


def compute_spread_zscore(
    spread: pd.Series,
    *,
    window: int = 60,
    ddof: int = 1,
) -> pd.Series:
    """Rolling z-score of the spread residual; no bfill."""
    return rolling_zscore(spread, window=window, ddof=ddof)


def compute_ewm_zscore(spread: pd.Series, *, span: int = 60) -> pd.Series:
    """EWM z-score of the spread residual; no bfill."""
    return ewm_zscore(spread, span=span)


def compute_ou_residual_score(spread: pd.Series, *, window: int = 60) -> pd.Series:
    """Rolling OU / AR(1) residual score of the spread."""
    return ou_residual_score(spread, window=window)


def compute_adaptive_zscore(
    spread: pd.Series,
    half_life: pd.Series,
    *,
    z_min: int = 20,
    z_max: int = 120,
) -> pd.Series:
    """Trad z with lagged half-life window; see ``adaptive_zscore``."""
    return adaptive_zscore(spread, half_life, z_min=z_min, z_max=z_max)


def compute_half_life(spread: pd.Series, *, window: int = 252) -> pd.Series:
    """Rolling discrete OU half-life of the spread (PIT series for gates / time-stops)."""
    return rolling_ou_half_life(spread, window=window)


def compute_coint_metrics(
    spread: pd.Series,
    *,
    metrics: Sequence[str] | None = None,
    adf_window: int = 252,
    adf_regression: str = "c",
    adf_autolag: str = "aic",
    var_window: int = 60,
    var_baseline_window: int = 252,
) -> pd.DataFrame:
    """Health metrics selector: ``adf_pvalue`` and/or ``variance_jump`` (``metrics`` like feature_subset)."""
    ids = resolve_feature_subset(
        metrics, COINT_METRICS, name="compute_coint_metrics"
    )
    cols: dict[str, pd.Series] = {}
    for mid in ids:
        if mid == "adf_pvalue":
            cols[mid] = rolling_adf_pvalue(
                spread,
                window=adf_window,
                regression=adf_regression,
                autolag=adf_autolag,
            )
        elif mid == "variance_jump":
            cols[mid] = residual_variance_ratio(
                spread,
                window=var_window,
                baseline_window=var_baseline_window,
            )
    return pd.DataFrame(cols, index=spread.index)


def run_cointegration_test(
    y_price: pd.Series,
    x_price: pd.Series,
    *,
    pvalue_threshold: float = COINT_PVALUE,
    trend: str = "c",
    autolag: str = "aic",
) -> CointResult:
    """Discovery-only Engle-Granger on log prices; never call inside a backtest loop."""
    _validate_pair_inputs(y_price, x_price)
    return test_cointegration(
        to_log_price(y_price),
        to_log_price(x_price),
        pvalue_threshold=pvalue_threshold,
        trend=trend,
        autolag=autolag,
    )


def screen_pair_cointegration(
    prices_by_ticker: Mapping[str, pd.Series],
    candidates: Sequence[tuple[str, str]],
    *,
    is_end: pd.Timestamp | str,
    ols_window: int = 252,
    min_is_bars: int | None = None,
    pvalue_threshold: float = COINT_PVALUE,
    lookback_bars: int | None = None,
) -> pd.DataFrame:
    """Engle-Granger screen on dates through calendar research-IS end ``is_end``.

    Caller passes a pre-registered universe cutoff ``is_end``. A pair is
    **ineligible** (not an EG fail) when mutual bars with ``date <= is_end`` are
    fewer than ``min_is_bars`` (default ``min(ols_window, 252)``): NaN metrics,
    ``eligible=False``. Eligible pairs run EG + discovery half-life on that IS
    slice only. ``pair_id`` / ``ticker_y`` / ``ticker_x`` follow EG direction
    (``y|x``). Discovery half-life is a scalar on the IS rolling-OLS spread
    (not used for panel z).

    ``lookback_bars`` restricts the screen to the last N mutual bars ending at
    ``is_end`` (trailing window) instead of all history through ``is_end``. Used by the
    rotating book, which re-screens each quarter on ``L`` trailing bars; the frozen book
    leaves it None and screens the full IS. Note the default ``min_is_bars`` of
    ``min(ols_window, 252)`` equals a 252-bar ``L``, so a rotating candidate must have the
    full trailing window present to be eligible.
    """
    is_end_ts = pd.Timestamp(is_end)
    if lookback_bars is not None and int(lookback_bars) < 2:
        raise ValueError("lookback_bars must be >= 2")
    floor = min(ols_window, 252) if min_is_bars is None else int(min_is_bars)
    if floor < 2:
        raise ValueError("min_is_bars must be >= 2")
    rows: list[dict] = []

    def _ineligible_row(
        ticker_a: str, ticker_b: str, *, n_is: int
    ) -> dict:
        return {
            "pair_id": f"{ticker_a}|{ticker_b}",
            "ticker_y": ticker_a,
            "ticker_x": ticker_b,
            "pvalue": float("nan"),
            "tstat": float("nan"),
            "discovery_half_life": float("nan"),
            "n_is_bars": n_is,
            "is_end": is_end_ts,
            "eligible": False,
        }

    for ticker_a, ticker_b in candidates:
        pa, pb = _align_pair_closes(prices_by_ticker, ticker_a, ticker_b)
        if pa.empty:
            rows.append(_ineligible_row(ticker_a, ticker_b, n_is=0))
            continue

        mask = pa.index <= is_end_ts
        y_is = pa.loc[mask]
        x_is = pb.loc[mask]
        if lookback_bars is not None:
            y_is = y_is.iloc[-int(lookback_bars) :]
            x_is = x_is.iloc[-int(lookback_bars) :]
        n_is = int(len(y_is))
        if n_is < floor:
            rows.append(_ineligible_row(ticker_a, ticker_b, n_is=n_is))
            continue

        result = run_cointegration_test(
            y_is, x_is, pvalue_threshold=pvalue_threshold
        )
        if result.direction == "x~y":
            ticker_y, ticker_x = ticker_b, ticker_a
            y_px, x_px = x_is, y_is
        else:
            ticker_y, ticker_x = ticker_a, ticker_b
            y_px, x_px = y_is, x_is

        hedge = compute_static_hedge_spread(y_px, x_px, window=ols_window)
        hl = ou_half_life(hedge["spread"].dropna())

        rows.append(
            {
                "pair_id": f"{ticker_y}|{ticker_x}",
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                "pvalue": float(result.pvalue),
                "tstat": float(result.tstat),
                "discovery_half_life": float(hl),
                "n_is_bars": n_is,
                "is_end": is_end_ts,
                "eligible": True,
            }
        )

    cols = [
        "pair_id",
        "ticker_y",
        "ticker_x",
        "pvalue",
        "tstat",
        "discovery_half_life",
        "n_is_bars",
        "is_end",
        "eligible",
    ]
    return pd.DataFrame(rows, columns=cols)


def build_pair_panel(
    ohlc_by_ticker: Mapping[str, pd.DataFrame],
    pairs: Sequence[tuple[str, str]],
    *,
    ols_window: int = 252,
    z_window: int = 60,
    hl_window: int = 252,
    include_adf_pvalue: bool = False,
    include_variance_jump: bool = False,
    hedge: str = "ols",
    kalman_delta: float = 1e-4,
    kalman_obs_var: float = 1e-3,
    kalman_burn_in: int = 30,
) -> pd.DataFrame:
    """Long panel of PIT hedge, fixed-window z, and rolling half-life.

    ``pairs`` must already be oriented as ``(ticker_y, ticker_x)``. Each value in
    ``ohlc_by_ticker`` is a DatetimeIndex frame with columns
    ``open`` / ``high`` / ``low`` / ``close``.

    ``hedge`` is ``ols`` (rolling window) or ``kalman`` (prior-state KF).
    Hedge, z, half-life, and optional ADF / variance-jump metrics use **closes
    only** (``close_y`` / ``close_x`` at bar ``t``). Open / high / low are stored
    for execution-path consumers (fill at next-bar open; high/low stops) and are
    never inputs to β / spread / z. Do not materialize next-bar open on the
    signal row — fill for a signal on ``date = t`` is the next pair-bar open.

    Fetching and candidate selection happen outside this store.
    """
    if hedge not in {"ols", "kalman"}:
        raise ValueError(f"hedge must be 'ols' or 'kalman', got {hedge!r}")

    metric_ids: list[str] = []
    if include_adf_pvalue:
        metric_ids.append("adf_pvalue")
    if include_variance_jump:
        metric_ids.append("variance_jump")

    extra_cols: list[str] = []
    if hedge == "kalman":
        extra_cols.extend(["spread_var", "z_innov"])

    frames: list[pd.DataFrame] = []
    for ticker_y, ticker_x in pairs:
        y_ohlc, x_ohlc = _align_pair_ohlc(ohlc_by_ticker, ticker_y, ticker_x)
        if y_ohlc.empty:
            continue

        y_close = y_ohlc["close"]
        x_close = x_ohlc["close"]
        if hedge == "kalman":
            hedge_df = compute_kalman_hedge_spread(
                y_close,
                x_close,
                delta=kalman_delta,
                obs_var=kalman_obs_var,
                burn_in=kalman_burn_in,
            )
        else:
            hedge_df = compute_static_hedge_spread(y_close, x_close, window=ols_window)
        z = compute_spread_zscore(hedge_df["spread"], window=z_window)
        hl = compute_half_life(hedge_df["spread"], window=hl_window)

        frame = pd.DataFrame(
            {
                "date": y_ohlc.index,
                "pair_id": f"{ticker_y}|{ticker_x}",
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                "open_y": y_ohlc["open"].to_numpy(dtype=float),
                "high_y": y_ohlc["high"].to_numpy(dtype=float),
                "low_y": y_ohlc["low"].to_numpy(dtype=float),
                "close_y": y_close.to_numpy(dtype=float),
                "open_x": x_ohlc["open"].to_numpy(dtype=float),
                "high_x": x_ohlc["high"].to_numpy(dtype=float),
                "low_x": x_ohlc["low"].to_numpy(dtype=float),
                "close_x": x_close.to_numpy(dtype=float),
                "alpha": hedge_df["alpha"].to_numpy(dtype=float),
                "beta": hedge_df["beta"].to_numpy(dtype=float),
                "spread": hedge_df["spread"].to_numpy(dtype=float),
                "z": z.to_numpy(dtype=float),
                "half_life": hl.to_numpy(dtype=float),
            }
        )
        if hedge == "kalman":
            frame["spread_var"] = hedge_df["spread_var"].to_numpy(dtype=float)
            frame["z_innov"] = hedge_df["z_innov"].to_numpy(dtype=float)
        if metric_ids:
            metrics = compute_coint_metrics(hedge_df["spread"], metrics=metric_ids)
            for col in metric_ids:
                frame[col] = metrics[col].to_numpy(dtype=float)
        frames.append(frame)

    if not frames:
        empty_cols = list(_PANEL_BASE_COLS) + extra_cols + metric_ids
        return pd.DataFrame(columns=empty_cols)

    out = pd.concat(frames, ignore_index=True)
    # Pair frames must already share one clock (naive UTC from fetch_ohlcv /
    # long_ohlcv_to_frames). Mixed HK/Tokyo tz-aware stamps fail here.
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "pair_id"], kind="mergesort").reset_index(drop=True)
