"""Download point-in-time top-N S&P 500 equities via yfinance."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import date, datetime

import pandas as pd
import yfinance as yf

from data.ingestion.sp500_universe import constituents_on_or_before
from data.repo_paths import data_cache_dir

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_YF_FIELDS = frozenset({"Open", "High", "Low", "Close", "Volume"})
CHUNK_SIZE = 50
MAX_RETRIES = 5
RETRY_DELAY_SEC = 3.0

DEFAULT_CACHE_DIR = data_cache_dir()


def _to_timestamp(value: date | str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _ranking_end(start_date: pd.Timestamp) -> pd.Timestamp:
    """Last business day on or before start_date (calendar approximation)."""
    ts = start_date
    while ts.weekday() >= 5:
        ts -= pd.Timedelta(days=1)
    return ts


VALID_OHLCV_INTERVALS = frozenset({"1d", "1h"})


def _cache_key(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    label: str,
    *,
    interval: str = "1d",
) -> str:
    payload = (
        f"{label}|{interval}|{start.date()}|{end.date()}|{','.join(sorted(tickers))}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _read_cache(cache_dir: str, key: str) -> pd.DataFrame | None:
    path = os.path.join(cache_dir, f"{key}.parquet")
    if not os.path.exists(path):
        return None
    cached = pd.read_parquet(path)
    # Empty frames are usually transient Yahoo misses — do not keep poisoning.
    if cached.empty:
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    logger.debug("Cache hit: %s", os.path.basename(path))
    return cached


def _write_cache(cache_dir: str, key: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.parquet")
    df.to_parquet(path, index=False)


# Yahoo exchange suffixes that must keep their dot (Asia / international listings).
_ASIAN_EXCHANGE_SUFFIXES = frozenset(
    {
        ".HK",
        ".T",
        ".L",
        ".TO",
        ".AX",
        ".NS",
        ".BO",
        ".SS",
        ".SZ",
        ".KS",
        ".KQ",
        ".TW",
        ".SI",
        ".JK",
        ".KL",
        ".BK",
    }
)


def _has_asian_exchange_suffix(symbol: str) -> bool:
    upper = symbol.upper()
    return any(upper.endswith(suf) for suf in _ASIAN_EXCHANGE_SUFFIXES)


def _canonical_to_yahoo(symbol: str, *, isAsian: bool = False) -> str:
    """Map project ticker form to Yahoo Finance symbol (e.g. ``BRK.B`` → ``BRK-B``).

    When ``isAsian=True``, known exchange suffixes (``.HK``, ``.T``, …) are kept
    so ``0700.HK`` / ``8306.T`` are not mangled into ``0700-HK`` / ``8306-T``.
    """
    if isAsian and _has_asian_exchange_suffix(symbol):
        return symbol
    return symbol.replace(".", "-")


def _yahoo_to_canonical(symbol: str, *, isAsian: bool = False) -> str:
    """Map Yahoo Finance symbol back to project ticker form (e.g. ``BRK-B`` → ``BRK.B``)."""
    if isAsian and _has_asian_exchange_suffix(symbol):
        return symbol
    return symbol.replace("-", ".")


def _build_yahoo_ticker_maps(
    tickers: list[str],
    *,
    isAsian: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Return Yahoo download symbols and a Yahoo→canonical lookup."""
    yahoo_tickers = [_canonical_to_yahoo(t, isAsian=isAsian) for t in tickers]
    yahoo_to_canonical = {y: c for y, c in zip(yahoo_tickers, tickers)}
    return yahoo_tickers, yahoo_to_canonical


def _wide_to_long(
    raw: pd.DataFrame,
    tickers: list[str],
    *,
    yahoo_to_canonical: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize yfinance output to long OHLCV format."""
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])

    if yahoo_to_canonical is None:
        yahoo_to_canonical = {t: t for t in tickers}

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        grouped_by_ticker = level0.isin(tickers).any()
        frames: list[pd.DataFrame] = []

        for ticker in tickers:
            canonical = yahoo_to_canonical.get(ticker, ticker)
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
            sub["ticker"] = canonical
            frames.append(sub[["date", "ticker", *OHLCV_COLUMNS]])

        if not frames:
            return pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])
        out = pd.concat(frames, ignore_index=True)
        out.columns.name = None
        return out

    ticker = tickers[0]
    canonical = yahoo_to_canonical.get(ticker, ticker)
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
    flat["ticker"] = canonical
    out = flat[["date", "ticker", *OHLCV_COLUMNS]]
    out.columns.name = None
    return out


def _download_ohlcv(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_dir: str | None,
    cache_label: str,
    isAsian: bool = False,
    interval: str = "1d",
) -> pd.DataFrame:
    """Batch-download OHLCV for tickers in [start, end] at ``interval`` (``1d`` or ``1h``)."""
    if interval not in VALID_OHLCV_INTERVALS:
        raise ValueError(
            f"interval must be one of {sorted(VALID_OHLCV_INTERVALS)}, got {interval!r}"
        )
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])

    if cache_dir is not None:
        key = _cache_key(tickers, start, end, cache_label, interval=interval)
        cached = _read_cache(cache_dir, key)
        if cached is not None:
            cached.columns.name = None
            return cached

    yahoo_tickers, yahoo_to_canonical = _build_yahoo_ticker_maps(
        tickers, isAsian=isAsian
    )

    # yfinance end is exclusive
    end_exclusive = end + pd.Timedelta(days=1)
    frames: list[pd.DataFrame] = []

    for i in range(0, len(yahoo_tickers), CHUNK_SIZE):
        chunk = yahoo_tickers[i : i + CHUNK_SIZE]
        raw: pd.DataFrame | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = yf.download(
                    chunk,
                    start=start.date(),
                    end=end_exclusive.date(),
                    interval=interval,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    logger.warning("yfinance failed for chunk %s: %s", chunk[:3], exc)
                    raw = pd.DataFrame()
                else:
                    time.sleep(RETRY_DELAY_SEC * attempt)
                    continue

            if raw is not None and not raw.empty:
                break
            if attempt < MAX_RETRIES:
                logger.warning(
                    "yfinance returned empty for chunk %s (attempt %d/%d)",
                    chunk[:3],
                    attempt,
                    MAX_RETRIES,
                )
                time.sleep(RETRY_DELAY_SEC * attempt)
                raw = pd.DataFrame()

        if raw is None or raw.empty:
            continue

        chunk_df = _wide_to_long(raw, chunk, yahoo_to_canonical=yahoo_to_canonical)
        if not chunk_df.empty:
            frames.append(chunk_df)

    if not frames:
        result = pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])
    else:
        result = pd.concat(frames, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        if interval == "1d":
            result["date"] = result["date"].dt.normalize()
        result.columns.name = None
        ohlcv = list(OHLCV_COLUMNS)
        result = result.loc[~result[ohlcv].isna().all(axis=1)].copy()

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
    cache_dir: str | None = DEFAULT_CACHE_DIR,
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
    panel.columns.name = None
    return panel


def fetch_ohlcv(
    ticker: str,
    start_date: str | date,
    end_date: str | date | None = None,
    *,
    cache_dir: str | None = DEFAULT_CACHE_DIR,
    isAsian: bool = False,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Return long-format OHLCV for a single ticker over ``[start_date, end_date]``.

    Columns: date, ticker, open, high, low, close, volume

    ``interval`` is ``1d`` (default) or ``1h``. Hourly timestamps are **not**
    normalized to midnight. This sleeve does not fetch 4-hour bars.

    Set ``isAsian=True`` for Yahoo exchange-suffixed names (``.HK``, ``.T``, …)
    so the suffix dot is not treated as a US share-class separator.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("ticker must be a non-empty string")

    start_ts = _to_timestamp(start_date)
    end_ts = _to_timestamp(end_date) if end_date is not None else _to_timestamp(datetime.now().date())

    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    panel = _download_ohlcv(
        [symbol],
        start_ts,
        end_ts,
        cache_dir=cache_dir,
        cache_label=f"single_{symbol}",
        isAsian=isAsian,
        interval=interval,
    )

    if panel.empty:
        return pd.DataFrame(columns=["date", "ticker", *OHLCV_COLUMNS])

    ohlcv = list(OHLCV_COLUMNS)
    all_nan = panel[ohlcv].isna().all(axis=1)
    panel = panel.loc[~all_nan].copy()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    panel.columns.name = None
    return panel
