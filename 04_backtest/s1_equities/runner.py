"""Run S1 long/short backtests for configured timing modes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.s1_equities.costs import AlpacaCostModel, DEFAULT_COSTS
from backtest.s1_equities.portfolio import (
    WEIGHT_EQUAL,
    WEIGHT_INV_VOL,
    VALID_WEIGHT_MODES,
    build_entry_weights,
)
from backtest.s1_equities.signals import (
    TIMING_MON_OPEN_FRI_CLOSE,
    TIMING_MON_OPEN_MON_OPEN,
    VALID_TIMING_MODES,
)
from backtest.s1_equities.stops import (
    STOP_ATR,
    STOP_PCT,
    StopConfig,
    VALID_STOP_MODES,
    apply_stops_over_hold,
    stop_levels_from_entry,
    stop_levels_from_entry_pct,
)
from backtest.s1_equities.timing import build_hold_table
from data.processing.feature_implementation.atr import wilder_atr_matrix
from data.processing.feature_implementation.realized_vol import (
    DEFAULT_OPEN_VOL_WINDOW,
    trailing_open_vol_matrix,
)
from risk.s1_equities.signal_conviction import (
    ICScaleConfig,
    ic_multiplier_from_history,
    ic_posterior_from_history,
)
from risk.s1_equities.vol_targeting import (
    ESTIMATOR_BAYES,
    ESTIMATOR_EWM,
    ESTIMATOR_ROLLING,
    VolTargetConfig,
    initial_bayes_vol_state,
    leverage_from_history,
    update_bayes_vol_state,
    vol_from_bayes_state,
)


@dataclass
class BacktestResult:
    """Period returns and diagnostics for one backtest configuration."""

    timing_mode: str
    n: int
    weight_mode: str
    stop_label: str
    period_returns: pd.Series  # indexed by entry_date; net of costs (levered)
    period_returns_gross: pd.Series
    turnover: pd.Series  # one-way turnover fraction (entry + early exits + schedule)
    entry_weights: pd.DataFrame  # base (unlevered) dollar-neutral weights
    hold_table: pd.DataFrame
    equity: pd.Series
    equity_gross: pd.Series
    stop_hit_rate: float  # fraction of name-holds that stopped early
    mean_names_stopped: float  # mean # names stopped per entry period
    mean_early_exit_notional: float  # mean sum |w| stopped per period (levered)
    leverage: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    vol_estimate: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ic_posterior: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ic_multiplier: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    period_returns_base: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    gross_exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _aligned_asset_return(
    weights_row: pd.Series,
    entry_px: pd.Series,
    exit_px: pd.Series,
) -> float:
    """Sum w * (exit/entry - 1) over names with finite weights and prices."""
    w = weights_row.replace(0.0, np.nan).dropna()
    if w.empty:
        return 0.0
    e = entry_px.reindex(w.index)
    x = exit_px.reindex(w.index)
    ok = e.notna() & x.notna() & (e > 0) & np.isfinite(e) & np.isfinite(x)
    if not ok.any():
        return 0.0
    w = w.loc[ok]
    r = x.loc[ok] / e.loc[ok] - 1.0
    return float((w * r).sum())


def _turnover_l1(prev: pd.Series | None, curr: pd.Series) -> float:
    """One-way turnover = 0.5 * L1 change in weights (full book if prev is None)."""
    if prev is None:
        return float(0.5 * curr.abs().sum())
    aligned = pd.concat([prev, curr], axis=1, keys=["p", "c"]).fillna(0.0)
    return float(0.5 * (aligned["c"] - aligned["p"]).abs().sum())


def _pit_lag_periods(timing_mode: str) -> int:
    """
    Number of most-recent completed periods to exclude from sizing history.

    Mon→Mon: the prior hold exits at this Monday open → exclude 1.
    Mon→Fri: prior hold already exited Friday → exclude 0.
    """
    if timing_mode == TIMING_MON_OPEN_MON_OPEN:
        return 1
    if timing_mode == TIMING_MON_OPEN_FRI_CLOSE:
        return 0
    raise ValueError(f"unknown timing_mode={timing_mode!r}")


def run_backtest(
    scores: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    n: int,
    timing_mode: str,
    weight_mode: str = WEIGHT_EQUAL,
    costs: AlpacaCostModel = DEFAULT_COSTS,
    date_mask: pd.Series | None = None,
    highs: pd.DataFrame | None = None,
    lows: pd.DataFrame | None = None,
    stop: StopConfig | None = None,
    inv_vol_window: int = DEFAULT_OPEN_VOL_WINDOW,
    vol_target: VolTargetConfig | None = None,
    ic_scale: ICScaleConfig | None = None,
    ic_inputs: pd.DataFrame | None = None,
    max_gross: float = 1.50,
) -> BacktestResult:
    """
    Dollar-neutral top/bottom-N backtest with optional stops and exposure overlays.

    Parameters
    ----------
    scores :
        Week-start score matrix (index=entry dates).
    opens, closes :
        Daily OHLC pivots (full calendar).
    highs, lows :
        Required when ``stop.mode`` is ``atr`` or ``pct`` for daily path checks.
    weight_mode :
        Portfolio weighting within sleeves.
    stop :
        Optional ``StopConfig`` (default: no stops).
    inv_vol_window :
        Trailing open-to-open vol window when ``weight_mode == 'inv_vol'``
        (default ``DEFAULT_OPEN_VOL_WINDOW`` = 21).
    date_mask :
        Optional boolean Series; only entry dates where True are traded.
    vol_target :
        Optional ``VolTargetConfig``; scales gross via ``risk.s1_equities.vol_targeting``.
    ic_scale :
        Optional ``ICScaleConfig``; multiplies vol leverage via IC conviction.
    ic_inputs :
        Optional DataFrame indexed by entry date with columns ``ic``, ``n_names``.
    max_gross :
        Hard cap on ``L_vol * m_ic`` (default 1.50).
    """
    if timing_mode not in VALID_TIMING_MODES:
        raise ValueError(f"unknown timing_mode={timing_mode!r}")
    if weight_mode not in VALID_WEIGHT_MODES:
        raise ValueError(f"unknown weight_mode={weight_mode!r}")
    if inv_vol_window < 1:
        raise ValueError("inv_vol_window must be >= 1")
    if max_gross <= 0:
        raise ValueError("max_gross must be positive")

    stop_cfg = stop if stop is not None else StopConfig()
    if stop_cfg.mode not in VALID_STOP_MODES:
        raise ValueError(f"unknown stop.mode={stop_cfg.mode!r}")
    use_atr = stop_cfg.mode == STOP_ATR
    use_pct = stop_cfg.mode == STOP_PCT
    use_stops = use_atr or use_pct
    if use_atr:
        if stop_cfg.atr_window is None or stop_cfg.atr_multiple is None:
            raise ValueError("atr stop requires atr_window and atr_multiple")
        if highs is None or lows is None:
            raise ValueError("atr stops require highs and lows panels")
    if use_pct:
        if stop_cfg.pct is None or not np.isfinite(stop_cfg.pct) or stop_cfg.pct <= 0:
            raise ValueError("pct stop requires positive pct")
        if highs is None or lows is None:
            raise ValueError("pct stops require highs and lows panels")

    vt_cfg = vol_target
    use_vt = vt_cfg is not None and vt_cfg.enabled
    ic_cfg = ic_scale
    use_ic = ic_cfg is not None and ic_cfg.enabled
    if use_ic:
        if ic_inputs is None:
            raise ValueError("ic_scale requires ic_inputs with columns ic, n_names")
        need = {"ic", "n_names"}
        if not need.issubset(set(ic_inputs.columns)):
            raise ValueError("ic_inputs must contain columns 'ic' and 'n_names'")

    scores = scores.sort_index()
    if date_mask is not None:
        mask = date_mask.reindex(scores.index).fillna(False).astype(bool)
        active_scores = scores.loc[mask]
    else:
        active_scores = scores

    vol = None
    if weight_mode == WEIGHT_INV_VOL:
        vol = trailing_open_vol_matrix(
            opens,
            window=inv_vol_window,
            pit_shift=1,
        )

    entry_w = build_entry_weights(
        active_scores,
        n,
        weight_mode=weight_mode,
        vol=vol,
    )
    cal = opens.index.union(closes.index).sort_values().unique()
    if highs is not None:
        cal = pd.DatetimeIndex(cal).union(highs.index).unique().sort_values()
    if lows is not None:
        cal = pd.DatetimeIndex(cal).union(lows.index).unique().sort_values()
    holds = build_hold_table(entry_w, timing_mode, trading_calendar=cal)

    atr_mat = None
    if use_atr:
        assert highs is not None and lows is not None
        atr_mat = wilder_atr_matrix(
            highs,
            lows,
            closes,
            window=int(stop_cfg.atr_window),
            pit_shift=1,
        )

    open_c = costs.one_way_fraction(side="open")
    close_c = costs.one_way_fraction(side="close")
    pit_lag = _pit_lag_periods(timing_mode)

    net_rets: dict[pd.Timestamp, float] = {}
    gross_rets: dict[pd.Timestamp, float] = {}
    base_nets: dict[pd.Timestamp, float] = {}
    turns: dict[pd.Timestamp, float] = {}
    lev_map: dict[pd.Timestamp, float] = {}
    vol_map: dict[pd.Timestamp, float] = {}
    ic_post_map: dict[pd.Timestamp, float] = {}
    ic_mult_map: dict[pd.Timestamp, float] = {}
    gross_exp_map: dict[pd.Timestamp, float] = {}
    n_stopped_list: list[float] = []
    early_notional_list: list[float] = []
    n_name_holds = 0
    n_name_stops = 0

    # Chronological base-net history for vol targeting (append after each period)
    base_hist_dates: list[pd.Timestamp] = []
    base_hist_vals: list[float] = []

    prev_w: pd.Series | None = None
    prev_leverage: float | None = None

    ic_frame = None
    if use_ic:
        assert ic_inputs is not None
        ic_frame = ic_inputs.copy()
        ic_frame.index = pd.DatetimeIndex(pd.to_datetime(ic_frame.index))
        ic_frame = ic_frame.sort_index()

    for _, row in holds.iterrows():
        entry = pd.Timestamp(row["entry_date"])
        exit_ = pd.Timestamp(row["exit_date"])
        w = entry_w.loc[entry]
        active = w.replace(0.0, np.nan).dropna()
        n_name_holds += int(len(active))

        if entry not in opens.index:
            continue

        entry_open = opens.loc[entry]

        if timing_mode == TIMING_MON_OPEN_MON_OPEN:
            if exit_ not in opens.index:
                continue
            schedule_exit = opens.loc[exit_]
        else:
            if exit_ not in closes.index:
                continue
            schedule_exit = closes.loc[exit_]

        # --- Exposure overlay (same scalar APIs as live) ---
        past_vals = base_hist_vals
        if pit_lag > 0 and len(past_vals) >= pit_lag:
            past_for_size = past_vals[:-pit_lag]
        else:
            past_for_size = past_vals if pit_lag == 0 else []

        l_vol = 1.0
        m_ic = 1.0
        ic_hat = np.nan
        vol_hat = np.nan

        if use_vt:
            assert vt_cfg is not None
            l_vol = leverage_from_history(
                past_for_size,
                vt_cfg,
                prev_leverage=prev_leverage,
            )
            # Diagnostic vol: recompute without deadband noise via series path
            if len(past_for_size) >= vt_cfg.min_periods:
                if vt_cfg.estimator == ESTIMATOR_BAYES:
                    st = initial_bayes_vol_state(vt_cfg)
                    for ri in past_for_size:
                        st = update_bayes_vol_state(st, float(ri), vt_cfg)
                    vol_hat = vol_from_bayes_state(st, vt_cfg)
                elif vt_cfg.estimator == ESTIMATOR_ROLLING:
                    arr = np.asarray(past_for_size, dtype=float)
                    windowed = arr[-int(vt_cfg.window) :]
                    if len(windowed) >= 2:
                        vol_hat = float(
                            np.std(windowed, ddof=1)
                            * np.sqrt(vt_cfg.periods_per_year)
                        )
                elif vt_cfg.estimator == ESTIMATOR_EWM:
                    lam = 0.5 ** (1.0 / float(vt_cfg.halflife))
                    sigma2 = np.nan
                    for ri in past_for_size:
                        if not np.isfinite(sigma2):
                            sigma2 = float(ri) ** 2
                        else:
                            sigma2 = lam * sigma2 + (1.0 - lam) * float(ri) ** 2
                    if np.isfinite(sigma2) and sigma2 > 0:
                        vol_hat = float(
                            np.sqrt(sigma2) * np.sqrt(vt_cfg.periods_per_year)
                        )

        if use_ic:
            assert ic_cfg is not None and ic_frame is not None
            # IC history with same pit lag: entries strictly before usable cutoff
            if base_hist_dates:
                usable_dates = (
                    base_hist_dates[:-pit_lag]
                    if pit_lag > 0
                    else list(base_hist_dates)
                )
            else:
                usable_dates = []
            # Prefer IC rows on/before those dates that exist in ic_frame
            if usable_dates:
                ic_hist = ic_frame.reindex(usable_dates).dropna(subset=["ic"])
            else:
                ic_hist = ic_frame.iloc[0:0]
            if len(ic_hist) > 0:
                m_ic = ic_multiplier_from_history(
                    ic_hist["ic"], ic_hist["n_names"], ic_cfg
                )
                ic_hat = ic_posterior_from_history(
                    ic_hist["ic"], ic_hist["n_names"], ic_cfg
                )
            else:
                m_ic = float(ic_cfg.warmup_multiplier)

        lo = float(vt_cfg.min_leverage) if use_vt and vt_cfg is not None else 0.25
        hi = float(max_gross)
        if use_vt and vt_cfg is not None:
            lo = float(vt_cfg.min_leverage)
            hi = min(float(vt_cfg.max_leverage), float(max_gross))
        lev = float(np.clip(l_vol * m_ic, lo, hi))
        if not (use_vt or use_ic):
            lev = 1.0
            lo, hi = 1.0, 1.0

        w_lev = w * lev

        if use_stops:
            assert highs is not None and lows is not None
            if use_atr:
                assert atr_mat is not None
                if entry not in atr_mat.index:
                    atr_row = pd.Series(np.nan, index=w.index, dtype=float)
                else:
                    atr_row = atr_mat.loc[entry]
                stops = stop_levels_from_entry(
                    entry_open,
                    w,
                    atr_row,
                    atr_multiple=float(stop_cfg.atr_multiple),
                )
            else:
                stops = stop_levels_from_entry_pct(
                    entry_open,
                    w,
                    pct=float(stop_cfg.pct),
                )
            exit_px, live_end_base, stopped = apply_stops_over_hold(
                w,
                entry=entry,
                exit_=exit_,
                entry_open=entry_open,
                stops=stops,
                opens=opens,
                highs=highs,
                lows=lows,
                schedule_exit_px=schedule_exit,
                exit_on_open=(timing_mode == TIMING_MON_OPEN_MON_OPEN),
            )
            gross = _aligned_asset_return(w_lev, entry_open, exit_px)
            early_notional = (
                float(w_lev.reindex(stopped).abs().sum()) if len(stopped) else 0.0
            )
            live_end = live_end_base * lev
            n_stopped_list.append(float(len(stopped)))
            early_notional_list.append(early_notional)
            n_name_stops += int(len(stopped))
        else:
            gross = _aligned_asset_return(w_lev, entry_open, schedule_exit)
            live_end = w_lev.copy()
            stopped = pd.Index([])
            early_notional = 0.0
            n_stopped_list.append(0.0)
            early_notional_list.append(0.0)

        # Base (unlevered) gross/net for vol-target history
        if use_stops:
            gross_base = _aligned_asset_return(w, entry_open, exit_px)
            early_base = (
                float(w.reindex(stopped).abs().sum()) if len(stopped) else 0.0
            )
        else:
            gross_base = _aligned_asset_return(w, entry_open, schedule_exit)
            early_base = 0.0

        if timing_mode == TIMING_MON_OPEN_MON_OPEN:
            to_entry = _turnover_l1(prev_w, w_lev)
            cost = to_entry * open_c + early_notional * close_c
            to = to_entry + early_notional
            if prev_w is None:
                to_entry_base = _turnover_l1(None, w)
            else:
                # prev_w is levered; recover approximate base by dividing last lev
                if prev_leverage is not None and abs(prev_leverage) > 1e-12:
                    prev_w_base = prev_w / prev_leverage
                else:
                    prev_w_base = prev_w
                to_entry_base = _turnover_l1(prev_w_base, w)
            cost_base = to_entry_base * open_c + early_base * close_c
            prev_w = live_end.copy()
        else:
            to_entry = _turnover_l1(None, w_lev)
            survivors = live_end.replace(0.0, np.nan).dropna()
            to_exit_surv = float(survivors.abs().sum())
            cost = (
                to_entry * open_c
                + early_notional * close_c
                + to_exit_surv * close_c
            )
            to = to_entry + early_notional + to_exit_surv
            to_entry_base = _turnover_l1(None, w)
            if use_stops:
                live_end_base = w.copy()
                live_end_base.loc[stopped] = 0.0
                survivors_base = live_end_base.replace(0.0, np.nan).dropna()
            else:
                survivors_base = w.replace(0.0, np.nan).dropna()
            to_exit_base = float(survivors_base.abs().sum())
            cost_base = (
                to_entry_base * open_c
                + early_base * close_c
                + to_exit_base * close_c
            )
            prev_w = None

        net = gross - cost
        net_base = gross_base - cost_base

        net_rets[entry] = net
        gross_rets[entry] = gross
        base_nets[entry] = net_base
        turns[entry] = to
        lev_map[entry] = lev
        vol_map[entry] = vol_hat
        ic_post_map[entry] = ic_hat
        ic_mult_map[entry] = m_ic
        gross_exp_map[entry] = float(w_lev.abs().sum())

        base_hist_dates.append(entry)
        base_hist_vals.append(net_base)
        prev_leverage = lev

    period_net = pd.Series(net_rets, dtype=float).sort_index()
    period_gross = pd.Series(gross_rets, dtype=float).sort_index()
    period_base = pd.Series(base_nets, dtype=float).sort_index()
    turnover = pd.Series(turns, dtype=float).sort_index()
    leverage = pd.Series(lev_map, dtype=float).sort_index()
    vol_estimate = pd.Series(vol_map, dtype=float).sort_index()
    ic_posterior = pd.Series(ic_post_map, dtype=float).sort_index()
    ic_multiplier = pd.Series(ic_mult_map, dtype=float).sort_index()
    gross_exposure = pd.Series(gross_exp_map, dtype=float).sort_index()

    if period_net.empty:
        equity = pd.Series(dtype=float)
        equity_gross = pd.Series(dtype=float)
    else:
        start = period_net.index[0] - pd.Timedelta(days=1)
        equity = pd.concat(
            [pd.Series([1.0], index=[start]), (1.0 + period_net).cumprod()]
        )
        equity_gross = pd.concat(
            [pd.Series([1.0], index=[start]), (1.0 + period_gross).cumprod()]
        )

    hit_rate = (
        float(n_name_stops) / float(n_name_holds) if n_name_holds > 0 else np.nan
    )
    mean_stopped = float(np.mean(n_stopped_list)) if n_stopped_list else np.nan
    mean_early = float(np.mean(early_notional_list)) if early_notional_list else np.nan

    return BacktestResult(
        timing_mode=timing_mode,
        n=n,
        weight_mode=weight_mode,
        stop_label=stop_cfg.label(),
        period_returns=period_net,
        period_returns_gross=period_gross,
        turnover=turnover,
        entry_weights=entry_w,
        hold_table=holds,
        equity=equity,
        equity_gross=equity_gross,
        stop_hit_rate=hit_rate,
        mean_names_stopped=mean_stopped,
        mean_early_exit_notional=mean_early,
        leverage=leverage,
        vol_estimate=vol_estimate,
        ic_posterior=ic_posterior,
        ic_multiplier=ic_multiplier,
        period_returns_base=period_base,
        gross_exposure=gross_exposure,
    )


def summarize_periods(
    period_returns: pd.Series,
    *,
    periods_per_year: float = 52.0,
) -> dict[str, float]:
    """Basic performance stats from period (weekly) net returns."""
    r = period_returns.dropna()
    if r.empty:
        return {
            "n_periods": 0,
            "total_return": np.nan,
            "avg_ann_return": np.nan,
            "cagr": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "mean_return": np.nan,
        }
    eq = (1.0 + r).cumprod()
    total = float(eq.iloc[-1] - 1.0)
    n = len(r)
    years = n / periods_per_year
    cagr = (
        float(eq.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and eq.iloc[-1] > 0
        else np.nan
    )
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else np.nan
    mu = float(r.mean() * periods_per_year)
    sharpe = mu / vol if vol and vol > 1e-12 else np.nan
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "n_periods": int(n),
        "total_return": total,
        "avg_ann_return": mu,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "win_rate": float((r > 0).mean()),
        "mean_return": float(r.mean()),
    }
