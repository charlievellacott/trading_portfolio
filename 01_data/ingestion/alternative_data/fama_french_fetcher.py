"""
ETF Tier A Carhart-style daily factors via ``fetch_ohlcv``.

Same public schema as the archived Ken French ZIP fetcher
(``02_research/notebooks/redundant/old_fama_french_fetcher.py``):
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

import pandas as pd

from data.ingestion.equity_fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

# alternative_data/ -> ingestion/ -> 01_data/ -> cache/
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "cache",
)

_CACHE_FILE = "etf_ff_factors_daily.parquet"

_ETF_TICKERS = ("SPY", "IWM", "IWD", "IWF", "MTUM", "BIL")

# MTUM listed ~2013-04; fetch from early 2013 by default
_DEFAULT_START = "2013-01-01"


def _simple_returns(close: pd.Series) -> pd.Series:
    return close.astype(float) / close.astype(float).shift(1) - 1.0


def _build_etf_factors(
    start_date: str,
    end_date: str | None,
    *,
    cache_dir: str,
) -> pd.DataFrame:
    """Fetch OHLCV for factor ETFs and build Carhart-style proxy columns."""
    frames: list[pd.DataFrame] = []
    for ticker in _ETF_TICKERS:
        panel = fetch_ohlcv(
            ticker,
            start_date,
            end_date,
            cache_dir=cache_dir,
        )
        if panel.empty:
            raise ValueError(f"No OHLCV returned for factor ETF {ticker!r}")
        slim = panel[["date", "close"]].copy()
        slim["date"] = pd.to_datetime(slim["date"])
        slim = slim.rename(columns={"close": ticker})
        frames.append(slim)

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

    if os.path.exists(cache_path):
        logger.debug("ETF FF cache hit: %s", cache_path)
        factors = pd.read_parquet(cache_path)
        factors["date"] = pd.to_datetime(factors["date"])
    else:
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
