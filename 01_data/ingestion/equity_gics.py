"""Yahoo Finance current GICS sector / industry labels for equity tickers."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from data.repo_paths import data_cache_dir

logger = logging.getLogger(__name__)

GICS_COLUMNS: tuple[str, ...] = ("ticker", "sector", "industry", "fetched_at")
DEFAULT_CACHE_NAME = "yahoo_gics.parquet"


def _default_cache_path() -> str:
    return os.path.join(data_cache_dir(), DEFAULT_CACHE_NAME)


def _normalize_label(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _fetch_one(ticker: str) -> dict:
    symbol = str(ticker).strip().upper()
    sector = ""
    industry = ""
    try:
        info = yf.Ticker(symbol).info or {}
        sector = _normalize_label(info.get("sector"))
        industry = _normalize_label(info.get("industry"))
    except Exception as exc:  # noqa: BLE001 — vendor failures stay missing
        logger.warning("Yahoo GICS fetch failed for %s: %s", symbol, exc)
    return {
        "ticker": symbol,
        "sector": sector,
        "industry": industry,
        "fetched_at": pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None),
    }


def _read_cache(path: str) -> pd.DataFrame:
    if not path or not os.path.isfile(path):
        return pd.DataFrame(columns=list(GICS_COLUMNS))
    df = pd.read_parquet(path)
    missing = [c for c in GICS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"GICS cache missing columns: {missing}")
    out = df.loc[:, list(GICS_COLUMNS)].copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["sector"] = out["sector"].map(_normalize_label)
    out["industry"] = out["industry"].map(_normalize_label)
    out["fetched_at"] = pd.to_datetime(out["fetched_at"])
    return out.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)


def _write_cache(path: str, frame: pd.DataFrame) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.loc[:, list(GICS_COLUMNS)].to_parquet(path, index=False)


def fetch_yahoo_gics(
    tickers: list[str] | tuple[str, ...] | pd.Series,
    *,
    cache_path: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return current Yahoo sector / industry for ``tickers``.

    Missing labels stay empty strings (never filled). Results are merged into
    a parquet cache (default ``01_data/cache/yahoo_gics.parquet``); tickers
    already present are skipped unless ``force_refresh`` is True.

    Classification is **current** GICS as served by Yahoo, not historical.
    """
    path = _default_cache_path() if cache_path is None else cache_path
    wanted = sorted({str(t).strip().upper() for t in list(tickers) if str(t).strip()})
    if not wanted:
        return pd.DataFrame(columns=list(GICS_COLUMNS))

    cached = _read_cache(path)
    if force_refresh:
        have: set[str] = set()
    else:
        have = set(cached["ticker"].astype(str)) if not cached.empty else set()

    to_fetch = [t for t in wanted if t not in have]
    rows: list[dict] = []
    for ticker in to_fetch:
        rows.append(_fetch_one(ticker))

    if rows:
        fresh = pd.DataFrame(rows, columns=list(GICS_COLUMNS))
        fresh["fetched_at"] = pd.to_datetime(fresh["fetched_at"])
        merged = pd.concat([cached, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=["ticker"], keep="last")
        _write_cache(path, merged)
    else:
        merged = cached

    out = merged.loc[merged["ticker"].isin(wanted), list(GICS_COLUMNS)].copy()
    # Preserve request order; missing (never fetched / failed) stay empty.
    by_ticker = {str(r.ticker): r for r in out.itertuples(index=False)}
    ordered: list[dict] = []
    for ticker in wanted:
        if ticker in by_ticker:
            row = by_ticker[ticker]
            ordered.append(
                {
                    "ticker": ticker,
                    "sector": _normalize_label(row.sector),
                    "industry": _normalize_label(row.industry),
                    "fetched_at": pd.Timestamp(row.fetched_at),
                }
            )
        else:
            ordered.append(
                {
                    "ticker": ticker,
                    "sector": "",
                    "industry": "",
                    "fetched_at": pd.NaT,
                }
            )
    return pd.DataFrame(ordered, columns=list(GICS_COLUMNS))
