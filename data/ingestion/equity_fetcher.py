"""Download point-in-time top-N S&P 500 equities via yfinance."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from data.ingestion.sp500_universe import constituents_on_or_before

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_YF_FIELDS = frozenset({"Open", "High", "Low", "Close", "Volume"})
CHUNK_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2.0

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def _to_timestamp(value: date | str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _ranking_end(start_date: pd.Timestamp) -> pd.Timestamp:
    """Last business day on or before start_date (calendar approximation)."""
    ts = start_date
    while ts.weekday() >= 5:
        ts -= pd.Timedelta(days=1)
    return ts


def _cache_key(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, label: str) -> str:
    payload = f"{label}|{start.date()}|{end.date()}|{','.join(sorted(tickers))}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _read_cache(cache_dir: Path, key: str) -> pd.DataFrame | None:
    path = cache_dir / f"{key}.parquet"
    if path.exists():
        logger.debug("Cache hit: %s", path.name)
        return pd.read_parquet(path)
    return None


def _write_cache(cache_dir: Path, key: str, df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.parquet"
    df.to_parquet(path, index=False)


def _wide_to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance output to long OHLCV format."""
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        grouped_by_ticker = level0.isin(tickers).any()
        frames: list[pd.DataFrame] = []

        for ticker in tickers:
            if grouped_by_ticker:
                if ticker not in level0:
                    continue
                sub = raw[ticker].copy()
            else:
                # group_by='column': (Open, AAPL), (High, AAPL), ...
                sub = pd.DataFrame(
                    {field: raw[(field, ticker)] for field in _YF_FIELDS if (field, ticker) in raw.columns}
                )
                if sub.empty:
                    continue

            sub = sub.reset_index()
            date_col = sub.columns[0]
            sub = sub.rename(
                columns={
                    date_col: "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            sub["ticker"] = ticker
            frames.append(sub[["date", "ticker", *OHLCV_COLUMNS]])

        if not frames:
            return pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])
        return pd.concat(frames, ignore_index=True)

    ticker = tickers[0]
    flat = raw.reset_index()
    date_col = flat.columns[0]
    flat = flat.rename(
        columns={
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    flat["ticker"] = ticker
    return flat[["date", "ticker", *OHLCV_COLUMNS]]


def _download_ohlcv(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_dir: Path | None,
    cache_label: str,
) -> pd.DataFrame:
    """Batch-download daily OHLCV for tickers in [start, end]."""
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])

    if cache_dir is not None:
        key = _cache_key(tickers, start, end, cache_label)
        cached = _read_cache(cache_dir, key)
        if cached is not None:
            return cached

    # yfinance end is exclusive
    end_exclusive = end + pd.Timedelta(days=1)
    frames: list[pd.DataFrame] = []

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        raw: pd.DataFrame | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = yf.download(
                    chunk,
                    start=start.date(),
                    end=end_exclusive.date(),
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    logger.warning("yfinance failed for chunk %s: %s", chunk[:3], exc)
                    raw = pd.DataFrame()
                else:
                    time.sleep(RETRY_DELAY_SEC * attempt)

        if raw is None or raw.empty:
            continue

        chunk_df = _wide_to_long(raw, chunk)
        if not chunk_df.empty:
            frames.append(chunk_df)

    if not frames:
        result = pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])
    else:
        result = pd.concat(frames, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()

    if cache_dir is not None:
        _write_cache(cache_dir, key, result)

    return result

## The ranking is done with 20 business days of data.
def _rank_by_dollar_volume(
    prices: pd.DataFrame,
    ranking_start: pd.Timestamp,
    ranking_end: pd.Timestamp,
    min_ranking_bars: int,
) -> pd.Series:
    """Average dollar volume per ticker over the ranking window."""
    window = prices[
        (prices["date"] >= ranking_start) & (prices["date"] <= ranking_end)
    ].copy()
    window["dollar_volume"] = window["close"] * window["volume"]

    stats = window.groupby("ticker").agg(
        avg_dollar_volume=("dollar_volume", "mean"),
        n_bars=("dollar_volume", "count"),
    )
    stats = stats[stats["n_bars"] >= min_ranking_bars]
    return stats["avg_dollar_volume"].sort_values(ascending=False)


def fetch_top_n_equities(
    n: int,
    start_date: str | date,
    *,
    lookback_days: int = 20,
    end_date: str | date | None = None,
    min_ranking_bars: int = 10,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Return long-format daily OHLCV for the top-n S&P 500 names by trailing
    dollar volume as of ``start_date``.

    Columns: date, ticker, open, high, low, close, volume

    Universe membership and ranking use only information on or before
    ``start_date`` (point-in-time S&P 500 snapshot + trailing volume window).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if min_ranking_bars < 1:
        raise ValueError("min_ranking_bars must be >= 1")

    start_ts = _to_timestamp(start_date)
    end_ts = _to_timestamp(end_date) if end_date is not None else _to_timestamp(datetime.now().date())

    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    universe_as_of, candidates = constituents_on_or_before(start_ts)
    logger.info(
        "PIT universe as of %s: %d candidates",
        universe_as_of.date(),
        len(candidates),
    )

    ranking_end = _ranking_end(start_ts)
    ranking_start = ranking_end - pd.offsets.BDay(lookback_days)

    ranking_prices = _download_ohlcv(
        candidates,
        ranking_start,
        ranking_end,
        cache_dir=cache_dir,
        cache_label="ranking",
    )

    dollar_volume = _rank_by_dollar_volume(
        ranking_prices,
        ranking_start,
        ranking_end,
        min_ranking_bars,
    )

    if dollar_volume.empty:
        raise ValueError(
            f"No tickers had at least {min_ranking_bars} ranking bars "
            f"between {ranking_start.date()} and {ranking_end.date()}."
        )

    top_tickers = dollar_volume.head(n).index.tolist()
    logger.info(
        "Selected top %d tickers by avg dollar volume (window %s to %s): %s",
        len(top_tickers),
        ranking_start.date(),
        ranking_end.date(),
        top_tickers,
    )

    panel = _download_ohlcv(
        top_tickers,
        start_ts,
        end_ts,
        cache_dir=cache_dir,
        cache_label="panel",
    )

    if panel.empty:
        return pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])

    ohlcv = list(OHLCV_COLUMNS)
    all_nan = panel[ohlcv].isna().all(axis=1)
    panel = panel.loc[~all_nan].copy()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel
