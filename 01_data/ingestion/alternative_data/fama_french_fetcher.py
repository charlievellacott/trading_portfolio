"""
ETF Tier A Carhart-style daily factors via ``fetch_ohlcv``.

Same public schema as the archived Ken French ZIP fetcher
(``02_research/notebooks/s1_equities/redundant/old_fama_french_fetcher.py``):
``date, mkt_rf, smb, hml, [mom,] rf`` in decimal daily *simple* returns.

These are free, live-updatable ETF proxies — not the academic Ken French
construction. Binding history constraint is MTUM (~2013).

Recipe
------
- ``rf``     = BIL close-to-close return
- ``mkt_rf`` = SPY − rf
- ``smb``    = IWM − SPY
- ``hml``    = IWD − IWF
- ``mom``    = MTUM − SPY
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import pandas as pd

from data.ingestion.equity_fetcher import (
    DEFAULT_CACHE_DIR,
    _download_ohlcv,
    _to_timestamp,
    fetch_ohlcv,
)

logger = logging.getLogger(__name__)

_CACHE_FILE = "etf_ff_factors_daily.parquet"

_ETF_TICKERS = ("SPY", "IWM", "IWD", "IWF", "MTUM", "BIL")

# MTUM listed ~2013-04; fetch from early 2013 by default
_DEFAULT_START = "2013-01-01"

# Pause between single-ticker fallbacks when Yahoo rate-limits.
_FETCH_PAUSE_SEC = 1.5
# When live end-date misses, probe recent calendar days for warm OHLCV caches.
_CACHE_END_LOOKBACK_DAYS = 7


def _simple_returns(close: pd.Series) -> pd.Series:
    return close.astype(float) / close.astype(float).shift(1) - 1.0


def _panels_from_long(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a long OHLCV panel into per-ticker date/close frames."""
    out: dict[str, pd.DataFrame] = {}
    if panel.empty:
        return out
    for ticker, grp in panel.groupby("ticker", sort=False):
        slim = grp[["date", "close"]].copy()
        slim["date"] = pd.to_datetime(slim["date"])
        slim = slim.dropna(subset=["close"])
        if slim.empty:
            continue
        slim = slim.rename(columns={"close": ticker})
        out[str(ticker)] = slim
    return out


def _fetch_etf_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str | None,
    *,
    cache_dir: str,
) -> pd.DataFrame:
    """Fetch one factor ETF; if open-ended end fails, reuse a recent warm cache."""
    panel = fetch_ohlcv(ticker, start_date, end_date, cache_dir=cache_dir)
    if not panel.empty or end_date is not None:
        return panel

    today = datetime.now().date()
    for back in range(1, _CACHE_END_LOOKBACK_DAYS + 1):
        probe_end = (pd.Timestamp(today) - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        panel = fetch_ohlcv(ticker, start_date, probe_end, cache_dir=cache_dir)
        if not panel.empty:
            logger.warning(
                "Using cached OHLCV for %s through %s (live end empty)",
                ticker,
                probe_end,
            )
            return panel
    return panel


def _build_etf_factors(
    start_date: str,
    end_date: str | None,
    *,
    cache_dir: str,
) -> pd.DataFrame:
    """Fetch OHLCV for factor ETFs and build Carhart-style proxy columns."""
    start_ts = _to_timestamp(start_date)
    end_ts = (
        _to_timestamp(end_date)
        if end_date is not None
        else _to_timestamp(datetime.now().date())
    )

    # One batch download — fewer Yahoo round-trips than six sequential calls.
    batch = _download_ohlcv(
        list(_ETF_TICKERS),
        start_ts,
        end_ts,
        cache_dir=cache_dir,
        cache_label="etf_ff_ohlcv",
    )
    by_ticker = _panels_from_long(batch)

    missing = [t for t in _ETF_TICKERS if t not in by_ticker]
    for i, ticker in enumerate(missing):
        if i:
            time.sleep(_FETCH_PAUSE_SEC)
        panel = _fetch_etf_ohlcv(
            ticker,
            start_date,
            end_date,
            cache_dir=cache_dir,
        )
        if panel.empty:
            raise ValueError(f"No OHLCV returned for factor ETF {ticker!r}")
        by_ticker.update(_panels_from_long(panel))

    still_missing = [t for t in _ETF_TICKERS if t not in by_ticker]
    if still_missing:
        raise ValueError(f"No OHLCV returned for factor ETF {still_missing[0]!r}")

    frames = [by_ticker[t] for t in _ETF_TICKERS]
    wide = frames[0]
    for other in frames[1:]:
        wide = wide.merge(other, on="date", how="inner")
    wide = wide.sort_values("date").reset_index(drop=True)

    rets = {t: _simple_returns(wide[t]) for t in _ETF_TICKERS}
    rf = rets["BIL"]
    spy = rets["SPY"]

    out = pd.DataFrame(
        {
            "date": wide["date"],
            "rf": rf,
            "mkt_rf": spy - rf,
            "smb": rets["IWM"] - spy,
            "hml": rets["IWD"] - rets["IWF"],
            "mom": rets["MTUM"] - spy,
        }
    )
    out = out.dropna(subset=["rf", "mkt_rf", "smb", "hml", "mom"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("ETF FF factor build produced no overlapping rows")
    return out


def fetch_ff_factors_daily(
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    include_momentum: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Fetch ETF-proxy daily factors (Carhart-style).

    Values are in decimal form (e.g. 0.01 = 1%). Cached to parquet under
    ``cache_dir`` as ``etf_ff_factors_daily.parquet``.

    Returns DataFrame with columns: date, mkt_rf, smb, hml, [mom,] rf.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _CACHE_FILE)

    factors: pd.DataFrame | None = None
    if os.path.exists(cache_path):
        factors = pd.read_parquet(cache_path)
        factors["date"] = pd.to_datetime(factors["date"])
        if factors.empty:
            logger.warning("Removing empty ETF FF cache: %s", cache_path)
            try:
                os.remove(cache_path)
            except OSError:
                pass
            factors = None

    if factors is None:
        fetch_start = _DEFAULT_START
        if start_date is not None and pd.Timestamp(start_date) < pd.Timestamp(fetch_start):
            fetch_start = str(pd.Timestamp(start_date).date())
        logger.info(
            "Building ETF FF factors via fetch_ohlcv (%s) ...",
            ", ".join(_ETF_TICKERS),
        )
        # Cache the full built series; apply start/end filters after load.
        factors = _build_etf_factors(fetch_start, None, cache_dir=cache_dir)
        factors.to_parquet(cache_path, index=False)
        logger.info("ETF FF cached: %s (%d rows)", cache_path, len(factors))

    if include_momentum:
        cols = ["date", "mkt_rf", "smb", "hml", "mom", "rf"]
    else:
        cols = ["date", "mkt_rf", "smb", "hml", "rf"]
    factors = factors[cols].copy()

    if start_date is not None:
        factors = factors[factors["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        factors = factors[factors["date"] <= pd.Timestamp(end_date)]

    return factors.sort_values("date").reset_index(drop=True)
