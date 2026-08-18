"""Run S2 pair-book backtests for research notebooks (not live Strategy)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data.processing.feature_implementation.hmm_regime import (
    GaussianHMM2Params,
    filter_mr_probability,
    fit_gaussian_hmm_2state,
)
from data.processing.s2_coint_store import (
    compute_adaptive_zscore,
    compute_ewm_zscore,
    compute_ou_residual_score,
)
from strategies.s2_coint.baseline import PERIODS_PER_YEAR
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.engine import BookSimResult, simulate_book
from strategies.s2_coint.metrics import corr_to_s1, metrics_from_returns
from strategies.s2_coint.spread_ohlc import attach_spread_indicators


@dataclass
class S2BacktestResult:
    config: S2SimConfig
    returns: pd.Series
    metrics: dict
    book: BookSimResult
    pair_trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def fit_hmm_on_train_dates(
    panel: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
) -> dict[str, GaussianHMM2Params]:
    """Fit a 2-state HMM on fold-train spread changes (H-013 v3)."""
    allowed = set(pd.DatetimeIndex(pd.to_datetime(train_dates)))
    out: dict[str, GaussianHMM2Params] = {}
    for pid, g in panel.groupby("pair_id", sort=False):
        g = g.sort_values("date")
        train = g.loc[pd.to_datetime(g["date"]).isin(allowed)]
        if "spread" not in train.columns or train.empty:
            continue
        chg = train["spread"].astype(float).diff()
        try:
            out[str(pid)] = fit_gaussian_hmm_2state(chg)
        except ValueError:
            continue
    return out


def periods_per_year_from_index(idx: pd.DatetimeIndex, *, bar: str) -> float:
    """Annualization from the return index; 1H uses median bars per calendar year."""
    if bar == "1d" or len(idx) < 2:
        return PERIODS_PER_YEAR
    s = pd.DatetimeIndex(pd.to_datetime(idx)).sort_values()
    years = s.to_series().dt.year
    counts = years.value_counts()
    if counts.empty:
        return PERIODS_PER_YEAR
    return float(counts.median())


def _prepare_panel(
    panel: pd.DataFrame,
    cfg: S2SimConfig,
    *,
    hmm_params: dict[str, GaussianHMM2Params] | None = None,
) -> pd.DataFrame:
    d = panel.copy()
    d["date"] = pd.to_datetime(d["date"])
    need_spread_ind = cfg.trend_mode != "off" or cfg.exit_mode == "hl3_atr_breaker"
    if need_spread_ind and "atr_spread" not in d.columns:
        d = attach_spread_indicators(
            d,
            atr_window=cfg.atr_window,
            include_rsi_adx=cfg.trend_mode != "off",
        )

    parts: list[pd.DataFrame] = []
    for pid, g in d.groupby("pair_id", sort=False):
        g = g.sort_values("date").copy()
        spread = g["spread"] if "spread" in g.columns else None
        if spread is not None and cfg.entry_mode == "v1_ewm_asym":
            g["z_ewm"] = compute_ewm_zscore(spread, span=cfg.z_window).to_numpy(dtype=float)
        if spread is not None and cfg.entry_mode == "v2_ou":
            g["ou_score"] = compute_ou_residual_score(spread, window=cfg.z_window).to_numpy(
                dtype=float
            )
        if spread is not None and cfg.z_window_mode in {"adaptive", "adaptive_alt"}:
            zmin, zmax = cfg.z_clip_min, cfg.z_clip_max
            if cfg.z_window_mode == "adaptive_alt":
                zmin, zmax = 10, 252
            g["z"] = compute_adaptive_zscore(
                spread, g["half_life"], z_min=zmin, z_max=zmax
            ).to_numpy(dtype=float)
        if cfg.entry_mode == "v3_hmm_innov" and hmm_params and pid in hmm_params:
            chg = spread.astype(float).diff() if spread is not None else None
            if chg is not None:
                g["p_mr"] = filter_mr_probability(chg, hmm_params[pid]).to_numpy(dtype=float)
        parts.append(g)
    if not parts:
        return d
    return pd.concat(parts, ignore_index=True)


def run_s2_backtest(
    panel: pd.DataFrame,
    cfg: S2SimConfig | None = None,
    *,
    date_mask: pd.Series | None = None,
    s1_weekly: pd.Series | None = None,
    mean_abs_score: float = 1.0,
    hmm_params: dict[str, GaussianHMM2Params] | None = None,
) -> S2BacktestResult:
    """Simulate the book on ``panel`` (optionally masked to val/OOS dates)."""
    cfg = cfg or S2SimConfig()
    d = _prepare_panel(panel, cfg, hmm_params=hmm_params)
    if date_mask is not None:
        dates = pd.to_datetime(d["date"])
        if date_mask.dtype == bool and len(date_mask) == len(d):
            d = d.loc[date_mask].copy()
        else:
            allowed = set(pd.DatetimeIndex(pd.to_datetime(date_mask)))
            d = d.loc[dates.isin(allowed)].copy()
    book = simulate_book(
        d,
        cfg,
        mean_abs_score=mean_abs_score,
        periods_per_year=periods_per_year_from_index(
            pd.DatetimeIndex(pd.to_datetime(d["date"])), bar=cfg.bar
        ),
    )
    ppy = periods_per_year_from_index(
        pd.DatetimeIndex(book.returns.index), bar=cfg.bar
    )
    m = metrics_from_returns(book.returns, periods_per_year=ppy)
    m["corr_to_s1"] = corr_to_s1(book.returns, s1_weekly)
    trades = []
    for res in book.pair_results.values():
        if not res.trades.empty:
            trades.append(res.trades)
    trade_df = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return S2BacktestResult(
        config=cfg,
        returns=book.returns,
        metrics=m,
        book=book,
        pair_trades=trade_df,
    )
