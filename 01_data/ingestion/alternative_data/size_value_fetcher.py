"""
Fetch daily size & valuation panels from SEC EDGAR Company Facts + OHLCV.

H-005 infrastructure: reconstructs a trading-day long panel of market_cap, pe,
and pb from filing-dated fundamentals (PIT on ``filed``) joined to daily
closes via ``merge_asof(..., direction='backward')``. Factor math does not
live here — callers merge the result onto an OHLCV panel on
``['date', 'ticker']``.

SEC requires a descriptive User-Agent. Default:
``trading_portfolio charlie.vellacott@gmail.com``. Override via the
``user_agent`` kwarg or the ``SEC_USER_AGENT`` environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from data.ingestion.equity_fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

DEFAULT_SEC_USER_AGENT = "trading_portfolio charlie.vellacott@gmail.com"

# alternative_data/ -> ingestion/ -> 01_data/ -> cache/
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "cache",
)

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
_TICKER_MAP_CACHE = "sec_company_tickers.json"
_FACTS_CACHE_SUBDIR = "sec_companyfacts"
_REQUEST_SLEEP_SEC = 0.12  # stay under SEC ~10 req/s guidance

_SHARE_TAGS: tuple[tuple[str, str], ...] = (
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
)
_BOOK_TAGS: tuple[tuple[str, str], ...] = (
    ("us-gaap", "StockholdersEquity"),
    ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
)
_EPS_TAGS: tuple[tuple[str, str], ...] = (
    ("us-gaap", "EarningsPerShareDiluted"),
    ("us-gaap", "EarningsPerShareBasic"),
)
_NET_INCOME_TAGS: tuple[tuple[str, str], ...] = (
    ("us-gaap", "NetIncomeLoss"),
    ("us-gaap", "ProfitLoss"),
)

_OUTPUT_COLS = (
    "date",
    "ticker",
    "shares_outstanding",
    "book_equity",
    "eps_ttm",
    "market_cap",
    "pe",
    "pb",
)


def _resolve_user_agent(user_agent: str | None) -> str:
    if user_agent is not None:
        ua = user_agent.strip()
        if not ua:
            raise ValueError(
                "user_agent must be a non-empty string "
                "(SEC requires a descriptive User-Agent)"
            )
        return ua
    env = os.environ.get("SEC_USER_AGENT")
    if env is not None:
        ua = env.strip()
        if not ua:
            raise ValueError(
                "SEC_USER_AGENT is set but empty; unset it or provide a "
                "non-empty value"
            )
        return ua
    return DEFAULT_SEC_USER_AGENT


def _sec_get(url: str, *, user_agent: str) -> requests.Response:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    time.sleep(_REQUEST_SLEEP_SEC)
    return resp


def _canonical_sec_ticker(symbol: str) -> str:
    """Map project ticker form to SEC ticker form (e.g. ``BRK.B`` → ``BRK-B``)."""
    return symbol.strip().upper().replace(".", "-")


def _cik10(cik: int | str) -> str:
    return str(int(cik)).zfill(10)


def _load_ticker_to_cik(
    cache_dir: str,
    *,
    user_agent: str,
) -> dict[str, str]:
    """Return SEC ticker (upper) → zero-padded CIK string."""
    path = os.path.join(cache_dir, _TICKER_MAP_CACHE)
    if os.path.exists(path):
        logger.debug("SEC ticker map cache hit: %s", path)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        logger.info("Downloading SEC company ticker map...")
        resp = _sec_get(_TICKER_MAP_URL, user_agent=user_agent)
        raw = resp.json()
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        logger.info("SEC ticker map cached: %s (%d entries)", path, len(raw))

    out: dict[str, str] = {}
    for entry in raw.values():
        ticker = str(entry.get("ticker", "")).strip().upper()
        cik = entry.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = _cik10(cik)
    return out


def _facts_cache_path(cache_dir: str, cik10: str) -> str:
    return os.path.join(cache_dir, _FACTS_CACHE_SUBDIR, f"{cik10}.json")


def _load_companyfacts(
    cik10: str,
    cache_dir: str,
    *,
    user_agent: str,
) -> dict[str, Any]:
    path = _facts_cache_path(cache_dir, cik10)
    if os.path.exists(path):
        logger.debug("CompanyFacts cache hit: %s", path)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    url = _COMPANYFACTS_URL.format(cik10=cik10)
    logger.info("Downloading CompanyFacts for CIK %s...", cik10)
    resp = _sec_get(url, user_agent=user_agent)
    data = resp.json()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info("CompanyFacts cached: %s", path)
    return data


def _concept_observations(
    facts: dict[str, Any],
    taxonomy: str,
    tag: str,
) -> list[dict[str, Any]]:
    try:
        units = facts["facts"][taxonomy][tag]["units"]
    except (KeyError, TypeError):
        return []

    rows: list[dict[str, Any]] = []
    for unit_key, entries in units.items():
        for entry in entries:
            filed = entry.get("filed")
            val = entry.get("val")
            if filed is None or val is None:
                continue
            rows.append(
                {
                    "filed": filed,
                    "val": val,
                    "end": entry.get("end"),
                    "fp": entry.get("fp"),
                    "form": entry.get("form"),
                    "unit": unit_key,
                }
            )
    return rows


def _first_tag_frame(
    facts: dict[str, Any],
    tags: tuple[tuple[str, str], ...],
    *,
    value_col: str,
) -> pd.DataFrame:
    """
    Return a DataFrame of filing-dated values for the first tag that has data.

    Columns: date (filed), ``value_col``. Duplicate filed dates keep the last
    observation after sorting by period end.
    """
    for taxonomy, tag in tags:
        rows = _concept_observations(facts, taxonomy, tag)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["filed"])
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df[value_col] = pd.to_numeric(df["val"], errors="coerce")
        df = df.dropna(subset=["date", value_col])
        df = df.sort_values(["date", "end"]).drop_duplicates("date", keep="last")
        return df[["date", value_col]].reset_index(drop=True)
    return pd.DataFrame(columns=["date", value_col])


def _eps_ttm_from_net_income(
    facts: dict[str, Any],
    shares: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build trailing-four-quarter EPS from quarterly NetIncomeLoss / shares.

    Uses filing dates for PIT. Returns columns date, eps_ttm.
    """
    ni = _first_tag_frame(facts, _NET_INCOME_TAGS, value_col="net_income")
    if ni.empty or shares.empty:
        return pd.DataFrame(columns=["date", "eps_ttm"])

    # Prefer 10-Q / quarterly-looking rows via fp when available from raw facts
    rows: list[dict[str, Any]] = []
    for taxonomy, tag in _NET_INCOME_TAGS:
        for obs in _concept_observations(facts, taxonomy, tag):
            fp = str(obs.get("fp") or "")
            form = str(obs.get("form") or "")
            # Keep quarterly frames; skip pure annual FY when Q frames exist later
            if fp in ("Q1", "Q2", "Q3", "Q4") or form in ("10-Q", "10-Q/A"):
                rows.append(obs)
            elif fp == "FY" or form in ("10-K", "10-K/A"):
                # Annual: treat as one observation; TTM of one annual ≈ annual EPS
                rows.append(obs)
        if rows:
            break

    if not rows:
        return pd.DataFrame(columns=["date", "eps_ttm"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["filed"])
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["net_income"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["date", "net_income"])
    df = df.sort_values(["date", "end"]).drop_duplicates("date", keep="last")

    merged = pd.merge_asof(
        df.sort_values("date"),
        shares.sort_values("date"),
        on="date",
        direction="backward",
    )
    merged = merged.dropna(subset=["shares_outstanding"])
    merged = merged[merged["shares_outstanding"] > 0].copy()
    if merged.empty:
        return pd.DataFrame(columns=["date", "eps_ttm"])

    merged["q_eps"] = merged["net_income"] / merged["shares_outstanding"]
    # Rolling sum of last 4 quarterly EPS observations by filing order
    merged["eps_ttm"] = merged["q_eps"].rolling(4, min_periods=1).sum()
    return merged[["date", "eps_ttm"]].reset_index(drop=True)


def _eps_from_reported(
    facts: dict[str, Any],
) -> pd.DataFrame:
    """
    Prefer diluted/basic EPS facts; treat successive quarterly reports as a
    trailing sum (min_periods=1) labeled ``eps_ttm``.
    """
    for taxonomy, tag in _EPS_TAGS:
        rows = _concept_observations(facts, taxonomy, tag)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["filed"])
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df["eps"] = pd.to_numeric(df["val"], errors="coerce")
        df = df.dropna(subset=["date", "eps"])
        df = df.sort_values(["date", "end"]).drop_duplicates("date", keep="last")
        # Prefer quarterly forms when present
        qmask = df["fp"].isin(["Q1", "Q2", "Q3", "Q4"]) | df["form"].isin(
            ["10-Q", "10-Q/A"]
        )
        if qmask.any():
            use = df.loc[qmask].copy()
            use["eps_ttm"] = use["eps"].rolling(4, min_periods=1).sum()
        else:
            use = df.copy()
            use["eps_ttm"] = use["eps"]
        return use[["date", "eps_ttm"]].reset_index(drop=True)
    return pd.DataFrame(columns=["date", "eps_ttm"])


def _extract_fundamentals(facts: dict[str, Any]) -> pd.DataFrame:
    """
    Collapse CompanyFacts into a filing-dated fundamentals frame.

    Columns: date, shares_outstanding, book_equity, eps_ttm.
    """
    shares = _first_tag_frame(facts, _SHARE_TAGS, value_col="shares_outstanding")
    book = _first_tag_frame(facts, _BOOK_TAGS, value_col="book_equity")
    eps = _eps_from_reported(facts)
    if eps.empty:
        eps = _eps_ttm_from_net_income(facts, shares)

    frames = [f for f in (shares, book, eps) if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=["date", "shares_outstanding", "book_equity", "eps_ttm"]
        )

    out = frames[0]
    for frame in frames[1:]:
        out = pd.merge(out, frame, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def _price_panel_for_tickers(
    tickers: list[str],
    start_date: str | date | None,
    end_date: str | date | None,
    *,
    price_panel: pd.DataFrame | None,
    cache_dir: str,
) -> pd.DataFrame:
    if price_panel is not None:
        required = {"date", "ticker", "close"}
        missing = required - set(price_panel.columns)
        if missing:
            raise ValueError(
                f"price_panel missing required columns: {sorted(missing)}"
            )
        panel = price_panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        panel["ticker"] = panel["ticker"].astype(str).str.strip().str.upper()
        ticker_set = {t.strip().upper() for t in tickers}
        panel = panel[panel["ticker"].isin(ticker_set)].copy()
        if start_date is not None:
            panel = panel[panel["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            panel = panel[panel["date"] <= pd.Timestamp(end_date)]
        return panel[["date", "ticker", "close"]].sort_values(
            ["ticker", "date"]
        ).reset_index(drop=True)

    if start_date is None:
        raise ValueError("start_date is required when price_panel is not supplied")

    end = end_date if end_date is not None else datetime.now().date()
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        frame = fetch_ohlcv(
            ticker,
            start_date,
            end,
            cache_dir=cache_dir,
        )
        if not frame.empty:
            frames.append(frame[["date", "ticker", "close"]])
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str).str.strip().str.upper()
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def _align_fundamentals_to_prices(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Per-ticker merge_asof of filing-dated fundamentals onto daily closes."""
    if prices.empty:
        return pd.DataFrame(columns=list(_OUTPUT_COLS))

    empty_fund = pd.DataFrame(
        columns=["date", "shares_outstanding", "book_equity", "eps_ttm"]
    )
    if fundamentals is None or fundamentals.empty:
        fundamentals = empty_fund

    pieces: list[pd.DataFrame] = []
    for ticker, grp in prices.groupby("ticker", sort=False):
        left = grp.sort_values("date")[["date", "ticker", "close"]].copy()
        right = fundamentals.sort_values("date").copy()
        if right.empty:
            merged = left.copy()
            merged["shares_outstanding"] = float("nan")
            merged["book_equity"] = float("nan")
            merged["eps_ttm"] = float("nan")
        else:
            merged = pd.merge_asof(
                left,
                right,
                on="date",
                direction="backward",
            )
        pieces.append(merged)

    out = pd.concat(pieces, ignore_index=True)
    out["market_cap"] = out["close"] * out["shares_outstanding"]
    out["pb"] = out["market_cap"] / out["book_equity"]
    out.loc[out["book_equity"].isna() | (out["book_equity"] <= 0), "pb"] = float(
        "nan"
    )
    out["pe"] = out["close"] / out["eps_ttm"]
    out.loc[out["eps_ttm"].isna() | (out["eps_ttm"] <= 0), "pe"] = float("nan")
    out.loc[out["shares_outstanding"].isna(), "market_cap"] = float("nan")
    return out[list(_OUTPUT_COLS)].sort_values(["date", "ticker"]).reset_index(
        drop=True
    )


def fetch_size_value_daily(
    tickers: list[str],
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    *,
    price_panel: pd.DataFrame | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """
    Fetch a daily long panel of size & valuation fields for ``tickers``.

    Reconstructs trading-day ``market_cap``, ``pe``, and ``pb`` from SEC
    Company Facts (PIT on filing date) and daily closes. Returns columns:
    date, ticker, shares_outstanding, book_equity, eps_ttm, market_cap, pe, pb.

    Join onto an OHLCV panel with::

        panel.merge(sv, on=["date", "ticker"], how="left")

    ``merge_asof`` is applied only inside this function. Pass ``price_panel``
    when OHLCV is already loaded to avoid re-downloading.

    User-Agent resolution: ``user_agent`` kwarg → ``SEC_USER_AGENT`` env →
    ``DEFAULT_SEC_USER_AGENT``.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list")

    ua = _resolve_user_agent(user_agent)
    os.makedirs(cache_dir, exist_ok=True)

    canonical = [t.strip().upper() for t in tickers]
    if any(not t for t in canonical):
        raise ValueError("tickers must not contain empty strings")

    prices = _price_panel_for_tickers(
        canonical,
        start_date,
        end_date,
        price_panel=price_panel,
        cache_dir=cache_dir,
    )
    if prices.empty:
        return pd.DataFrame(columns=list(_OUTPUT_COLS))

    ticker_to_cik = _load_ticker_to_cik(cache_dir, user_agent=ua)

    pieces: list[pd.DataFrame] = []
    for ticker in sorted(prices["ticker"].unique()):
        sec_ticker = _canonical_sec_ticker(ticker)
        cik10 = ticker_to_cik.get(sec_ticker)
        if cik10 is None:
            # Also try without hyphenization change (already upper)
            cik10 = ticker_to_cik.get(ticker)
        if cik10 is None:
            logger.warning("No SEC CIK for ticker %s; leaving fundamentals NaN", ticker)
            fund = pd.DataFrame(
                columns=["date", "shares_outstanding", "book_equity", "eps_ttm"]
            )
        else:
            try:
                facts = _load_companyfacts(cik10, cache_dir, user_agent=ua)
                fund = _extract_fundamentals(facts)
            except requests.HTTPError as exc:
                logger.warning(
                    "CompanyFacts fetch failed for %s (CIK %s): %s",
                    ticker,
                    cik10,
                    exc,
                )
                fund = pd.DataFrame(
                    columns=["date", "shares_outstanding", "book_equity", "eps_ttm"]
                )

        ticker_prices = prices[prices["ticker"] == ticker]
        pieces.append(_align_fundamentals_to_prices(ticker_prices, fund))

    if not pieces:
        return pd.DataFrame(columns=list(_OUTPUT_COLS))
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
