"""
BIS real effective exchange rate (REER) helpers via FRED mirrors.

``BIS_REER_SERIES`` maps ISO currency → FRED series id (BIS broad/narrow
REER mirrors commonly published on FRED)::

    USD RBUSBIS, EUR RBXMBIS, JPY RBJPBIS, GBP RBGBBIS,
    CHF RBCHBIS, CAD RBCABIS, AUD RBAUBIS, NZD RBNZBIS

Monthly prints are treated as known on ``availability_date`` = first day of
month M+2 (conservative publication lag). Pair value signal is
``-zscore(log(REER_base) - log(REER_quote))`` so a relatively rich (high)
base REER leans **short** the base (negative signal for a long-base position).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from data.ingestion.alternative_data.fred_fetcher import fetch_fred_series

logger = logging.getLogger(__name__)

BIS_REER_SERIES: dict[str, str] = {
    "USD": "RBUSBIS",
    "EUR": "RBXMBIS",
    "JPY": "RBJPBIS",
    "GBP": "RBGBBIS",
    "CHF": "RBCHBIS",
    "CAD": "RBCABIS",
    "AUD": "RBAUBIS",
    "NZD": "RBNZBIS",
}


def _normalize_ccy(ccy: str) -> str:
    return str(ccy).strip().upper()


def _normalize_pair(pair: str) -> tuple[str, str]:
    p = str(pair).strip().upper().replace("/", "").replace("_", "").replace("-", "")
    if len(p) != 6:
        raise ValueError(f"expected 6-letter FX pair like EURUSD, got {pair!r}")
    return p[:3], p[3:]


def _month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx).to_period("M").to_timestamp("M"))


def fetch_bis_reer(
    currencies: Iterable[str],
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Monthly wide BIS REER frame (columns = currency).

    Index is month-end. Also includes ``availability_date`` = first calendar
    day of month M+2 (when the M print is assumed known).
    """
    cols: dict[str, pd.Series] = {}
    for ccy in currencies:
        currency = _normalize_ccy(ccy)
        series_id = BIS_REER_SERIES.get(currency)
        if series_id is None:
            raise KeyError(
                f"no BIS_REER_SERIES entry for {currency!r}; "
                f"known: {sorted(BIS_REER_SERIES)}"
            )
        s = fetch_fred_series(series_id, start=start, end=end)
        s.index = _month_end_index(pd.DatetimeIndex(s.index))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.name = currency
        cols[currency] = s

    if not cols:
        return pd.DataFrame()

    wide = pd.DataFrame(cols).sort_index()
    wide.index.name = "date"
    # availability: first day of month M+2 (e.g. Jan month-end → 1 Mar)
    wide = wide.copy()
    wide["availability_date"] = pd.DatetimeIndex(wide.index) + pd.offsets.MonthBegin(2)
    logger.info(
        "BIS REER frame currencies=%s rows=%d",
        list(cols.keys()),
        len(wide),
    )
    return wide


def pair_reer_value_signal(
    pair: str,
    reer_df: pd.DataFrame,
    *,
    z_window: int = 36,
) -> pd.Series:
    """
    Daily-aligned REER value signal for ``pair``.

    For a daily index, ``merge_asof`` on ``availability_date`` so only prints
    that could have been known are used. Signal::

        -zscore( log(REER_base) - log(REER_quote) )

    Higher relative REER for the base ⇒ lean short base (negative for long base).
    """
    base, quote = _normalize_pair(pair)
    if base not in reer_df.columns or quote not in reer_df.columns:
        raise KeyError(
            f"reer_df missing {base} and/or {quote}; columns={list(reer_df.columns)}"
        )
    if "availability_date" not in reer_df.columns:
        raise KeyError("reer_df must include availability_date column")

    slim = reer_df[[base, quote, "availability_date"]].dropna(subset=[base, quote]).copy()
    # Drop a named DatetimeIndex (often ``date``) so merge_asof on column
    # ``date`` is unambiguous after renaming ``availability_date``.
    slim = slim.reset_index(drop=True).sort_values("availability_date")
    log_spread = np.log(slim[base].astype(float)) - np.log(slim[quote].astype(float))
    slim["log_reer_spread"] = log_spread

    # Rolling z-score on monthly observations (window in months)
    mu = slim["log_reer_spread"].rolling(z_window, min_periods=max(6, z_window // 3)).mean()
    sd = slim["log_reer_spread"].rolling(z_window, min_periods=max(6, z_window // 3)).std()
    slim["signal_m"] = -(slim["log_reer_spread"] - mu) / sd.replace(0.0, np.nan)

    # Build a daily index spanning availability range and asof-merge
    start = pd.Timestamp(slim["availability_date"].min())
    end = pd.Timestamp(slim["availability_date"].max())
    daily = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    right = (
        slim[["availability_date", "signal_m"]]
        .rename(columns={"availability_date": "date"})
        .reset_index(drop=True)
        .sort_values("date")
    )
    merged = pd.merge_asof(
        daily.sort_values("date"),
        right,
        on="date",
        direction="backward",
    )
    out = pd.Series(
        merged["signal_m"].values,
        index=pd.DatetimeIndex(merged["date"], name="date"),
        name=f"{base}{quote}_reer_value",
    )
    return out
