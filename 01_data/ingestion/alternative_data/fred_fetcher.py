"""
FRED (St. Louis Fed) series fetcher with local parquet cache.

Auth via ``require_credential("FRED_API_KEY")`` from
``data.ingestion.credentials_env``. Placeholder ``keyhere`` / missing key
raises a clear ``ValueError``.

Cache layout: ``01_data/cache/fred/{series_id}.parquet``.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Iterable

import pandas as pd
import requests

from data.ingestion.credentials_env import require_credential
from data.repo_paths import data_cache_dir

logger = logging.getLogger(__name__)

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_CACHE_DIR = os.path.join(data_cache_dir(), "fred")
_USER_AGENT = (
    "Mozilla/5.0 (compatible; trading_portfolio/0.1; "
    "+https://github.com/local/trading_portfolio)"
)


def _to_date_str(value: date | str | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _cache_path(series_id: str, cache_dir: str | None = None) -> str:
    root = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    return os.path.join(root, f"{series_id}.parquet")


def _read_series_cache(path: str) -> pd.Series | None:
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("failed reading FRED cache %s: %s", path, exc)
        return None
    if df.empty:
        return None
    if "date" not in df.columns or "value" not in df.columns:
        return None
    s = pd.Series(
        df["value"].astype(float).values,
        index=pd.to_datetime(df["date"]),
        name=os.path.splitext(os.path.basename(path))[0],
    )
    s.index.name = "date"
    logger.debug("FRED cache hit: %s", os.path.basename(path))
    return s.sort_index()


def _write_series_cache(path: str, series: pd.Series) -> None:
    if series is None or series.empty:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(series.index),
            "value": series.astype(float).values,
        }
    )
    out.to_parquet(path, index=False)


def _download_fred_observations(
    series_id: str,
    *,
    api_key: str,
    start: str | None,
    end: str | None,
    frequency: str | None,
) -> pd.Series:
    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    if frequency:
        params["frequency"] = frequency

    headers = {"User-Agent": _USER_AGENT}
    logger.info("FRED download %s start=%s end=%s freq=%s", series_id, start, end, frequency)
    resp = requests.get(FRED_OBS_URL, params=params, headers=headers, timeout=60)
    if resp.status_code == 400 and "api_key" in resp.text.lower():
        raise ValueError(
            "FRED rejected the API key. Set a real FRED_API_KEY=... in "
            "config/credentials.env (placeholder 'keyhere' is not valid)."
        )
    resp.raise_for_status()
    payload = resp.json()
    obs = payload.get("observations") or []
    if not obs:
        logger.warning("FRED returned no observations for %s", series_id)
        return pd.Series(dtype=float, name=series_id)

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in obs:
        raw = row.get("value", ".")
        if raw is None or str(raw).strip() in {".", ""}:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
        dates.append(pd.Timestamp(row["date"]))

    s = pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name=series_id)
    return s.sort_index()


def fetch_fred_series(
    series_id: str,
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
    *,
    frequency: str | None = None,
    cache_dir: str | None = None,
    refresh: bool = False,
) -> pd.Series:
    """
    Fetch one FRED series as a ``pd.Series`` (DatetimeIndex, float values).

    Results are cached under ``01_data/cache/fred/{series_id}.parquet``.
    Date filters apply after cache load (full series is cached when possible).
    """
    sid = str(series_id).strip().upper()
    path = _cache_path(sid, cache_dir=cache_dir)
    series: pd.Series | None = None if refresh else _read_series_cache(path)

    if series is None:
        api_key = require_credential("FRED_API_KEY")
        series = _download_fred_observations(
            sid,
            api_key=api_key,
            start=None,
            end=None,
            frequency=frequency,
        )
        series.name = sid
        _write_series_cache(path, series)
    elif frequency is not None:
        # Cached raw series; caller asked for a frequency transform — refetch
        api_key = require_credential("FRED_API_KEY")
        series = _download_fred_observations(
            sid,
            api_key=api_key,
            start=_to_date_str(start),
            end=_to_date_str(end),
            frequency=frequency,
        )
        series.name = sid

    start_s = _to_date_str(start)
    end_s = _to_date_str(end)
    out = series.copy()
    if start_s is not None:
        out = out.loc[out.index >= pd.Timestamp(start_s)]
    if end_s is not None:
        out = out.loc[out.index <= pd.Timestamp(end_s)]
    out.name = sid
    return out


def fetch_fred_series_frame(
    series_ids: Iterable[str],
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
    *,
    frequency: str | None = None,
    cache_dir: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch multiple FRED series into a wide DataFrame (columns = series ids)."""
    cols: dict[str, pd.Series] = {}
    for sid in series_ids:
        s = fetch_fred_series(
            sid,
            start=start,
            end=end,
            frequency=frequency,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        cols[str(sid).strip().upper()] = s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()
