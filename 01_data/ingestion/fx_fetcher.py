"""FX OHLCV ingestion: OANDA v20 primary, yfinance daily fallback.

Daily bars use NY 17:00 day boundary via OANDA ``alignmentTimezone`` /
``dailyAlignment=17``. Timestamps are stored as naive UTC (S2 clock).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Iterable
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

from data.ingestion.credentials_env import read_credential, require_credential
from data.ingestion.equity_fetcher import ohlcv_dates_to_naive_utc
from data.repo_paths import data_cache_dir

logger = logging.getLogger(__name__)

G10_V1_PAIRS = (
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
)
NY_CLOSE_TZ = "America/New_York"
NY_CLOSE_HOUR = 17
VALID_INTERVALS = frozenset({"1d", "1h", "1m"})
VALID_PRICE_SIDES = frozenset({"M", "B", "A"})  # mid / bid / ask
OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"
OANDA_LIVE_URL = "https://api-fxtrade.oanda.com"
OANDA_CANDLE_LIMIT = 5000
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2.0

DEFAULT_CACHE_DIR = data_cache_dir()

_OANDA_GRANULARITY = {
    "1d": "D",
    "1h": "H1",
    "1m": "M1",
}

_YF_PAIR_SUFFIX = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
}


def _to_timestamp(value: date | str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _normalize_pair(pair: str) -> str:
    p = str(pair).strip().upper().replace("_", "").replace("/", "").replace("=X", "")
    if len(p) != 6:
        raise ValueError(f"expected 6-letter FX pair, got {pair!r}")
    return p


def _oanda_instrument(pair: str) -> str:
    p = _normalize_pair(pair)
    return f"{p[:3]}_{p[3:]}"


def _fx_cache_dir(cache_dir: str | None, source: str) -> str:
    root = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    path = os.path.join(root, "fx", source)
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(
    cache_dir: str | None,
    *,
    source: str,
    pair: str,
    interval: str,
    price_side: str,
) -> str:
    d = _fx_cache_dir(cache_dir, source)
    return os.path.join(d, f"{_normalize_pair(pair)}_{interval}_{price_side}.parquet")


def _read_pair_cache(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed reading FX cache %s: %s", path, exc)
        return None
    if df is None or df.empty:
        return None
    return df


def _write_pair_cache(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = df.copy()
    if "date" in out.columns:
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out.to_parquet(path, index=False)


def _oanda_base_url(env: str | None) -> str:
    e = (env or read_credential("OANDA_ACCOUNT_ENV") or "practice").strip().lower()
    if e in {"live", "trade", "fxtrade"}:
        return OANDA_LIVE_URL
    return OANDA_PRACTICE_URL


def _oanda_session(*, credentials_path: str | None = None) -> tuple[requests.Session, str]:
    token = require_credential("OANDA_API_TOKEN", path=credentials_path)
    env = read_credential("OANDA_ACCOUNT_ENV", path=credentials_path) or "practice"
    sess = requests.Session()
    sess.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept-Datetime-Format": "RFC3339",
            "Content-Type": "application/json",
        }
    )
    return sess, _oanda_base_url(env)


def _rfc3339(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _candle_to_row(c: dict, *, pair: str, source: str, price_side: str) -> dict | None:
    mid = c.get("mid") or c.get("bid") or c.get("ask")
    if not mid:
        return None
    return {
        "date": c.get("time"),
        "open": float(mid["o"]),
        "high": float(mid["h"]),
        "low": float(mid["l"]),
        "close": float(mid["c"]),
        "pair": pair,
        "source": source,
        "price_side": price_side,
        "complete": bool(c.get("complete", True)),
    }


def _oanda_candles(
    pair: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    interval: str = "1d",
    price: str = "M",
    day_boundary: str = "ny_1700",
    credentials_path: str | None = None,
) -> pd.DataFrame:
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be in {sorted(VALID_INTERVALS)}")
    price = str(price).upper()
    if price not in VALID_PRICE_SIDES:
        raise ValueError(f"price must be in {sorted(VALID_PRICE_SIDES)}")

    sess, base = _oanda_session(credentials_path=credentials_path)
    instrument = quote(_oanda_instrument(pair), safe="_")
    url = f"{base}/v3/instruments/{instrument}/candles"
    granularity = _OANDA_GRANULARITY[interval]
    price_param = {"M": "M", "B": "B", "A": "A"}[price]

    rows: list[dict] = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    while cursor < end_ts:
        params: dict[str, str | int] = {
            "granularity": granularity,
            "price": price_param,
            "from": _rfc3339(cursor),
            "to": _rfc3339(end_ts),
            "count": OANDA_CANDLE_LIMIT,
        }
        if interval == "1d" and day_boundary == "ny_1700":
            params["alignmentTimezone"] = NY_CLOSE_TZ
            params["dailyAlignment"] = NY_CLOSE_HOUR

        last_err: Exception | None = None
        data = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = sess.get(url, params=params, timeout=60)
                if resp.status_code == 429:
                    time.sleep(RETRY_DELAY_SEC * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(RETRY_DELAY_SEC * (attempt + 1))
        if data is None:
            raise RuntimeError(f"OANDA candles failed for {pair}: {last_err}")

        candles = data.get("candles") or []
        if not candles:
            break
        for c in candles:
            row = _candle_to_row(
                c, pair=_normalize_pair(pair), source="oanda", price_side=price
            )
            if row is not None:
                rows.append(row)
        last_time = pd.Timestamp(candles[-1]["time"])
        if last_time.tzinfo is None:
            last_time = last_time.tz_localize("UTC")
        nxt = last_time + pd.Timedelta(seconds=1)
        if nxt <= cursor:
            break
        cursor = nxt
        if len(candles) < OANDA_CANDLE_LIMIT:
            break

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "pair",
                "source",
                "price_side",
            ]
        )
    out = pd.DataFrame(rows)
    out["date"] = ohlcv_dates_to_naive_utc(out["date"])
    if interval == "1d":
        # Keep calendar day of the NY-aligned close (date part of UTC instant).
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def fetch_fx_ohlcv_yf(
    pair: str,
    start: date | str | pd.Timestamp,
    end: date | str | pd.Timestamp | None = None,
    *,
    interval: str = "1d",
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Yahoo Finance fallback (``EURUSD=X`` etc.). Daily preferred."""
    if interval not in {"1d", "1h"}:
        raise ValueError("yfinance FX fallback supports interval in {'1d','1h'}")
    p = _normalize_pair(pair)
    yf_sym = _YF_PAIR_SUFFIX.get(p)
    if yf_sym is None:
        yf_sym = f"{p}=X"
    start_ts = _to_timestamp(start)
    end_ts = _to_timestamp(end) if end is not None else _to_timestamp(datetime.now(timezone.utc).date())
    assert start_ts is not None and end_ts is not None

    path = _cache_path(
        cache_dir, source="yfinance", pair=p, interval=interval, price_side="M"
    )
    cached = _read_pair_cache(path)
    if cached is not None:
        mask = (cached["date"] >= start_ts.normalize()) & (
            cached["date"] <= end_ts.normalize() + pd.Timedelta(days=1)
        )
        hit = cached.loc[mask].copy()
        if not hit.empty:
            return hit.reset_index(drop=True)

    raw = yf.download(
        yf_sym,
        start=start_ts.strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "pair",
                "source",
                "price_side",
            ]
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.reset_index()
    date_col = "Datetime" if "Datetime" in raw.columns else "Date"
    out = pd.DataFrame(
        {
            "date": raw[date_col],
            "open": raw["Open"].astype(float),
            "high": raw["High"].astype(float),
            "low": raw["Low"].astype(float),
            "close": raw["Close"].astype(float),
            "pair": p,
            "source": "yfinance",
            "price_side": "M",
        }
    )
    if interval == "1h":
        out["date"] = ohlcv_dates_to_naive_utc(out["date"])
    else:
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    _write_pair_cache(path, out)
    return out.reset_index(drop=True)


def fetch_fx_ohlcv(
    pair: str,
    start: date | str | pd.Timestamp,
    end: date | str | pd.Timestamp | None = None,
    *,
    interval: str = "1d",
    source: str = "oanda",
    price: str = "M",
    day_boundary: str = "ny_1700",
    cache_dir: str | None = None,
    credentials_path: str | None = None,
) -> pd.DataFrame:
    """Fetch one FX pair OHLCV panel.

    Returns columns: ``date, open, high, low, close, pair, source, price_side``.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be in {sorted(VALID_INTERVALS)}")
    p = _normalize_pair(pair)
    start_ts = _to_timestamp(start)
    end_ts = _to_timestamp(end) if end is not None else _to_timestamp(
        datetime.now(timezone.utc).date()
    )
    assert start_ts is not None and end_ts is not None
    src = str(source).lower()

    if src == "yfinance":
        return fetch_fx_ohlcv_yf(
            p, start_ts, end_ts, interval=interval if interval != "1m" else "1h",
            cache_dir=cache_dir,
        )

    if src != "oanda":
        raise ValueError(f"unknown source {source!r}; expected 'oanda' or 'yfinance'")

    path = _cache_path(
        cache_dir, source="oanda", pair=p, interval=interval, price_side=price.upper()
    )
    cached = _read_pair_cache(path)
    need_fetch = True
    if cached is not None and not cached.empty:
        cmin = pd.Timestamp(cached["date"].min())
        cmax = pd.Timestamp(cached["date"].max())
        if cmin <= start_ts.normalize() and cmax >= (end_ts.normalize() - pd.Timedelta(days=2)):
            need_fetch = False

    if need_fetch:
        fetched = _oanda_candles(
            p,
            start_ts,
            end_ts + pd.Timedelta(days=1),
            interval=interval,
            price=price,
            day_boundary=day_boundary,
            credentials_path=credentials_path,
        )
        if cached is not None and not cached.empty:
            fetched = pd.concat([cached, fetched], ignore_index=True)
        if not fetched.empty:
            _write_pair_cache(path, fetched)
            cached = fetched

    if cached is None or cached.empty:
        logger.warning("OANDA empty for %s; falling back to yfinance", p)
        return fetch_fx_ohlcv_yf(p, start_ts, end_ts, interval="1d", cache_dir=cache_dir)

    mask = (cached["date"] >= start_ts.normalize()) & (
        cached["date"] <= end_ts.normalize() + pd.Timedelta(hours=23)
    )
    out = cached.loc[mask].copy()
    return out.reset_index(drop=True)


def fetch_fx_ohlcv_batch(
    pairs: Iterable[str],
    start: date | str | pd.Timestamp,
    end: date | str | pd.Timestamp | None = None,
    *,
    interval: str = "1d",
    source: str = "oanda",
    price: str = "M",
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Long-panel batch fetch over ``pairs``."""
    frames = [
        fetch_fx_ohlcv(
            p,
            start,
            end,
            interval=interval,
            source=source,
            price=price,
            cache_dir=cache_dir,
        )
        for p in pairs
    ]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "pair",
                "source",
                "price_side",
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values(["date", "pair"]).reset_index(
        drop=True
    )
