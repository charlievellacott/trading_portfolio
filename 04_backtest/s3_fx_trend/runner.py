"""Run S3 FX trend backtests for research notebooks (not live Strategy)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategies.s3_fx_trend.config import S3SimConfig
from strategies.s3_fx_trend.engine import BookSimResult, PERIODS_PER_YEAR, simulate_book
from strategies.s3_fx_trend.metrics import (
    corr_to_s1,
    corr_to_s2,
    cost_bps_per_year,
    metrics_from_returns,
    metrics_from_returns_inference,
)


@dataclass
class S3BacktestResult:
    config: S3SimConfig
    returns: pd.Series
    metrics: dict
    book: BookSimResult
    pair_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns_gross: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


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


def _allowed_dates_from_mask(panel: pd.DataFrame, date_mask: pd.Series) -> set:
    if pd.api.types.is_bool_dtype(date_mask):
        if len(date_mask) != len(panel):
            raise ValueError(
                f"boolean date_mask length {len(date_mask)} != panel rows {len(panel)}"
            )
        return set(pd.to_datetime(panel.loc[date_mask.astype(bool), "date"]))
    return set(pd.DatetimeIndex(pd.to_datetime(date_mask)))


def run_s3_backtest(
    panel: pd.DataFrame,
    cfg: S3SimConfig | None = None,
    *,
    date_mask: pd.Series | None = None,
    s1_weekly: pd.Series | None = None,
    s2_daily: pd.Series | None = None,
    rates_df: pd.DataFrame | None = None,
    reer_df: pd.DataFrame | None = None,
    n_trials_local: int | None = None,
    n_trials_stack: int | None = None,
) -> S3BacktestResult:
    """Simulate the S3 book on ``panel`` (optionally masked to val/OOS dates)."""
    cfg = cfg or S3SimConfig()
    d = panel.copy()
    d["date"] = pd.to_datetime(d["date"])
    if date_mask is not None:
        allowed = _allowed_dates_from_mask(panel, date_mask)
        d = d.loc[pd.to_datetime(d["date"]).isin(allowed)].copy()

    book = simulate_book(d, cfg, rates_df=rates_df, reer_df=reer_df)
    ppy = periods_per_year_from_index(
        pd.DatetimeIndex(book.returns_net.index), bar=str(cfg.bar)
    )
    m = metrics_from_returns_inference(
        book.returns_net,
        periods_per_year=ppy,
        n_trials_local=n_trials_local,
        n_trials_stack=n_trials_stack,
    )
    gross_m = metrics_from_returns(book.returns_gross, periods_per_year=ppy)
    m["ann_sharpe_gross"] = gross_m["ann_sharpe"]
    m["corr_to_s1"] = corr_to_s1(book.returns_net, s1_weekly)
    m["corr_to_s2"] = corr_to_s2(book.returns_net, s2_daily)
    trades = book.trades if book.trades is not None else pd.DataFrame()
    m["cost_bps_per_year"] = cost_bps_per_year(
        book.returns_net, trades, periods_per_year=ppy
    )
    n_entries = int(len(trades)) if trades is not None and not trades.empty else 0
    m["n_entries"] = n_entries

    return S3BacktestResult(
        config=cfg,
        returns=book.returns_net,
        metrics=m,
        book=book,
        pair_trades=trades.copy() if trades is not None else pd.DataFrame(),
        returns_gross=book.returns_gross,
    )
