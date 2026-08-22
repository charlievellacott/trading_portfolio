"""
SEC EDGAR Company Facts fetchers for size/value (H-005), gross profitability
(H-008), and filing-event clock anchors (H-010 supporting).

Shared infrastructure: User-Agent, ticker→CIK map, CompanyFacts JSON cache,
concept tag extraction, and filing-dated ``merge_asof(..., direction='backward')``
onto a daily price calendar (PIT on ``filed``, not period end).

Factor math does not live here — callers merge the result onto an OHLCV panel on
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

import numpy as np
import pandas as pd
import requests

from data.ingestion.equity_fetcher import fetch_ohlcv
from data.repo_paths import data_cache_dir

logger = logging.getLogger(__name__)

DEFAULT_SEC_USER_AGENT = "trading_portfolio charlie.vellacott@gmail.com"

DEFAULT_CACHE_DIR = data_cache_dir()

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
_TICKER_MAP_CACHE = "sec_company_tickers.json"
_FACTS_CACHE_SUBDIR = "sec_companyfacts"
_REQUEST_SLEEP_SEC = 0.12  # stay under SEC ~10 req/s guidance

# ---------------------------------------------------------------------------
# H-005 size / value tags
# ---------------------------------------------------------------------------
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

_SV_OUTPUT_COLS = (
    "date",
    "ticker",
    "shares_outstanding",
    "book_equity",
    "eps_ttm",
    "market_cap",
    "pe",
    "pb",
)
_SV_FUND_COLS = ("shares_outstanding", "book_equity", "eps_ttm")

# ---------------------------------------------------------------------------
# H-008 gross profitability tags
# ---------------------------------------------------------------------------
_GROSS_PROFIT_TAGS: tuple[tuple[str, str], ...] = (("us-gaap", "GrossProfit"),)
_REVENUE_TAGS: tuple[tuple[str, str], ...] = (
    ("us-gaap", "Revenues"),
    ("us-gaap", "SalesRevenueNet"),
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
)
_COGS_TAGS: tuple[tuple[str, str], ...] = (
    ("us-gaap", "CostOfGoodsAndServicesSold"),
    ("us-gaap", "CostOfRevenue"),
    ("us-gaap", "CostOfGoodsSold"),
)
# Prefer total Assets; AssetsCurrent is last-resort only (understates if used alone).
_ASSETS_TAGS: tuple[tuple[str, str], ...] = (
    ("us-gaap", "Assets"),
    ("us-gaap", "AssetsCurrent"),
)

_GP_OUTPUT_COLS = (
    "date",
    "ticker",
    "gross_profit_ttm",
    "assets",
    "gp_asset",
)
_GP_FUND_COLS = ("gross_profit_ttm", "assets")

# ---------------------------------------------------------------------------
# H-010 supporting · filing event clock
# ---------------------------------------------------------------------------
_FILING_FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A")
_FILING_CLOCK_OUTPUT_COLS = (
    "date",
    "ticker",
    "last_filed",
    "expected_next_filed",
)
_FILING_CLOCK_FUND_COLS = ("last_filed", "expected_next_filed")


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


def _quarterly_flow_ttm(
    facts: dict[str, Any],
    tags: tuple[tuple[str, str], ...],
    *,
    value_col: str,
    ttm_col: str,
) -> pd.DataFrame:
    """
    Trailing-four-quarter sum of a flow concept, labeled by filing date.

    Prefer quarterly ``fp`` / 10-Q rows when present; otherwise use all
    observations as-is (annual ≈ one TTM observation).
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
        qmask = df["fp"].isin(["Q1", "Q2", "Q3", "Q4"]) | df["form"].isin(
            ["10-Q", "10-Q/A"]
        )
        if qmask.any():
            use = df.loc[qmask].copy()
            use[ttm_col] = use[value_col].rolling(4, min_periods=1).sum()
        else:
            use = df.copy()
            use[ttm_col] = use[value_col]
        return use[["date", ttm_col]].reset_index(drop=True)
    return pd.DataFrame(columns=["date", ttm_col])


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

    rows: list[dict[str, Any]] = []
    for taxonomy, tag in _NET_INCOME_TAGS:
        for obs in _concept_observations(facts, taxonomy, tag):
            fp = str(obs.get("fp") or "")
            form = str(obs.get("form") or "")
            if fp in ("Q1", "Q2", "Q3", "Q4") or form in ("10-Q", "10-Q/A"):
                rows.append(obs)
            elif fp == "FY" or form in ("10-K", "10-K/A"):
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

    left = df.sort_values("date").copy()
    right = shares.sort_values("date").copy()
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    right["date"] = pd.to_datetime(right["date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
    )
    
    merged = merged.dropna(subset=["shares_outstanding"])
    merged = merged[merged["shares_outstanding"] > 0].copy()
    if merged.empty:
        return pd.DataFrame(columns=["date", "eps_ttm"])

    merged["q_eps"] = merged["net_income"] / merged["shares_outstanding"]
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
    Collapse CompanyFacts into a filing-dated size/value fundamentals frame.

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


def _distinct_filed_dates(
    facts: dict[str, Any],
    forms: tuple[str, ...],
) -> list[pd.Timestamp]:
    """
    Collect distinct filing dates across all CompanyFacts tags for ``forms``.

    Sorted ascending. Observations with missing ``filed`` or form outside
    ``forms`` are dropped.
    """
    form_set = set(forms)
    filed: set[pd.Timestamp] = set()
    taxonomies = facts.get("facts") or {}
    if not isinstance(taxonomies, dict):
        return []
    for taxonomy, tags in taxonomies.items():
        if not isinstance(tags, dict):
            continue
        for tag in tags:
            for obs in _concept_observations(facts, taxonomy, tag):
                if obs.get("form") not in form_set:
                    continue
                raw = obs.get("filed")
                if raw is None:
                    continue
                ts = pd.Timestamp(raw)
                if pd.isna(ts):
                    continue
                filed.add(ts.normalize())
    return sorted(filed)


def _forecast_expected_next_filed(
    filed: list[pd.Timestamp],
    i: int,
) -> pd.Timestamp:
    """
    PIT forecast of the next filing date using only events ``0..i``.

    Primary (``i >= 3``): same-slot last year — ``filed[i-3] + 365`` days.
    Fallback (``i >= 1``): ``filed[i]`` + median of the last up-to-4 gaps among
    events ``0..i``. Else ``NaT``.
    """
    if i >= 3:
        return filed[i - 3] + pd.Timedelta(days=365)
    if i >= 1:
        n_gaps = min(4, i)
        gaps = [
            (filed[j + 1] - filed[j]).days for j in range(i - n_gaps, i)
        ]
        return filed[i] + pd.Timedelta(days=float(np.median(gaps)))
    return pd.NaT


def _extract_filing_clock_events(
    facts: dict[str, Any],
    forms: tuple[str, ...],
) -> pd.DataFrame:
    """
    Event-level filing-clock frame: one row per distinct filed date.

    Columns: ``date`` (= filed), ``last_filed``, ``expected_next_filed``.
    ``expected_next_filed`` is forecast using only filings known at that event
    (no lookahead).
    """
    filed = _distinct_filed_dates(facts, forms)
    if not filed:
        return pd.DataFrame(
            columns=["date", "last_filed", "expected_next_filed"]
        )
    rows: list[dict[str, Any]] = []
    for i, ts in enumerate(filed):
        rows.append(
            {
                "date": ts,
                "last_filed": ts,
                "expected_next_filed": _forecast_expected_next_filed(filed, i),
            }
        )
    return pd.DataFrame(rows)


def _align_filing_clock_to_prices(
    prices: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Per-ticker merge_asof of filing-clock events onto the trading calendar."""
    if prices.empty:
        return pd.DataFrame(columns=list(_FILING_CLOCK_OUTPUT_COLS))

    out = _merge_asof_by_ticker(
        prices, events, fund_cols=_FILING_CLOCK_FUND_COLS
    )
    out["last_filed"] = pd.to_datetime(out["last_filed"], errors="coerce")
    out["expected_next_filed"] = pd.to_datetime(
        out["expected_next_filed"], errors="coerce"
    )
    return (
        out[list(_FILING_CLOCK_OUTPUT_COLS)]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def _extract_gp_fundamentals(facts: dict[str, Any]) -> pd.DataFrame:
    """
    Filing-dated TTM gross profit and assets.

    Prefer ``GrossProfit`` TTM via ``_quarterly_flow_ttm``; else
    ``Revenue_ttm − COGS_ttm``. Assets are a stock (latest filed), not summed.
    """
    gp_ttm = _quarterly_flow_ttm(
        facts,
        _GROSS_PROFIT_TAGS,
        value_col="gross_profit",
        ttm_col="gross_profit_ttm",
    )
    if gp_ttm.empty:
        rev_ttm = _quarterly_flow_ttm(
            facts, _REVENUE_TAGS, value_col="revenue", ttm_col="revenue_ttm"
        )
        cogs_ttm = _quarterly_flow_ttm(
            facts, _COGS_TAGS, value_col="cogs", ttm_col="cogs_ttm"
        )
        if not rev_ttm.empty and not cogs_ttm.empty:
            merged = pd.merge(rev_ttm, cogs_ttm, on="date", how="inner")
            merged["gross_profit_ttm"] = merged["revenue_ttm"] - merged["cogs_ttm"]
            gp_ttm = merged[["date", "gross_profit_ttm"]]
        else:
            gp_ttm = pd.DataFrame(columns=["date", "gross_profit_ttm"])

    assets = _first_tag_frame(facts, _ASSETS_TAGS, value_col="assets")

    frames = [f for f in (gp_ttm, assets) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "gross_profit_ttm", "assets"])

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


def _merge_asof_by_ticker(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    fund_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Per-ticker merge_asof of filing-dated fundamentals onto daily prices."""
    if prices.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", *fund_cols])

    empty_fund = pd.DataFrame(columns=["date", *fund_cols])
    if fundamentals is None or fundamentals.empty:
        fundamentals = empty_fund

    pieces: list[pd.DataFrame] = []
    for _ticker, grp in prices.groupby("ticker", sort=False):
        left = grp.sort_values("date")[["date", "ticker", "close"]].copy()
        right = fundamentals.sort_values("date").copy()
        if right.empty:
            merged = left.copy()
            for col in fund_cols:
                merged[col] = float("nan")
        else:
            keep = ["date", *[c for c in fund_cols if c in right.columns]]
            right = right[keep]
            for col in fund_cols:
                if col not in right.columns:
                    right[col] = float("nan")
            left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
            right["date"] = pd.to_datetime(right["date"]).astype("datetime64[ns]")
            merged = pd.merge_asof(
                left,
                right,
                on="date",
                direction="backward",
            )
        pieces.append(merged)

    return pd.concat(pieces, ignore_index=True)


def _align_fundamentals_to_prices(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Per-ticker merge_asof of size/value fundamentals onto daily closes."""
    if prices.empty:
        return pd.DataFrame(columns=list(_SV_OUTPUT_COLS))

    out = _merge_asof_by_ticker(prices, fundamentals, fund_cols=_SV_FUND_COLS)
    out["market_cap"] = out["close"] * out["shares_outstanding"]
    out["pb"] = out["market_cap"] / out["book_equity"]
    out.loc[out["book_equity"].isna() | (out["book_equity"] <= 0), "pb"] = float(
        "nan"
    )
    out["pe"] = out["close"] / out["eps_ttm"]
    out.loc[out["eps_ttm"].isna() | (out["eps_ttm"] <= 0), "pe"] = float("nan")
    out.loc[out["shares_outstanding"].isna(), "market_cap"] = float("nan")
    return out[list(_SV_OUTPUT_COLS)].sort_values(["date", "ticker"]).reset_index(
        drop=True
    )


def _align_gp_to_prices(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Per-ticker merge_asof of GP fundamentals; compute ``gp_asset``."""
    if prices.empty:
        return pd.DataFrame(columns=list(_GP_OUTPUT_COLS))

    out = _merge_asof_by_ticker(prices, fundamentals, fund_cols=_GP_FUND_COLS)
    out["gp_asset"] = out["gross_profit_ttm"] / out["assets"]
    out.loc[out["assets"].isna() | (out["assets"] <= 0), "gp_asset"] = float("nan")
    return out[list(_GP_OUTPUT_COLS)].sort_values(["date", "ticker"]).reset_index(
        drop=True
    )


def _lookup_cik(
    ticker: str,
    ticker_to_cik: dict[str, str],
) -> str | None:
    sec_ticker = _canonical_sec_ticker(ticker)
    cik10 = ticker_to_cik.get(sec_ticker)
    if cik10 is None:
        cik10 = ticker_to_cik.get(ticker)
    return cik10


def _validate_tickers(tickers: list[str]) -> list[str]:
    if not tickers:
        raise ValueError("tickers must be a non-empty list")
    canonical = [t.strip().upper() for t in tickers]
    if any(not t for t in canonical):
        raise ValueError("tickers must not contain empty strings")
    return canonical


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
    canonical = _validate_tickers(tickers)
    ua = _resolve_user_agent(user_agent)
    os.makedirs(cache_dir, exist_ok=True)

    prices = _price_panel_for_tickers(
        canonical,
        start_date,
        end_date,
        price_panel=price_panel,
        cache_dir=cache_dir,
    )
    if prices.empty:
        return pd.DataFrame(columns=list(_SV_OUTPUT_COLS))

    ticker_to_cik = _load_ticker_to_cik(cache_dir, user_agent=ua)
    empty_fund = pd.DataFrame(
        columns=["date", "shares_outstanding", "book_equity", "eps_ttm"]
    )

    pieces: list[pd.DataFrame] = []
    for ticker in sorted(prices["ticker"].unique()):
        cik10 = _lookup_cik(ticker, ticker_to_cik)
        if cik10 is None:
            logger.warning("No SEC CIK for ticker %s; leaving fundamentals NaN", ticker)
            fund = empty_fund
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
                fund = empty_fund

        ticker_prices = prices[prices["ticker"] == ticker]
        pieces.append(_align_fundamentals_to_prices(ticker_prices, fund))

    if not pieces:
        return pd.DataFrame(columns=list(_SV_OUTPUT_COLS))
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def fetch_gross_profitability_daily(
    tickers: list[str],
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    *,
    price_panel: pd.DataFrame | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """
    Fetch a daily long panel of TTM gross profitability for ``tickers``.

    Reconstructs trading-day ``gross_profit_ttm``, ``assets``, and
    ``gp_asset = gross_profit_ttm / assets`` from SEC Company Facts (PIT on
    filing date) joined onto the daily price calendar. Prefer ``GrossProfit``;
    else Revenue − COGS. Assets: ``Assets``, else ``AssetsCurrent`` (last
    resort). ``gp_asset`` is NaN when assets are missing or ``<= 0`` (no floor).

    Join onto an OHLCV panel with::

        panel.merge(gp, on=["date", "ticker"], how="left")

    Pass ``price_panel`` when OHLCV is already loaded to avoid re-downloading.
    """
    canonical = _validate_tickers(tickers)
    ua = _resolve_user_agent(user_agent)
    os.makedirs(cache_dir, exist_ok=True)

    prices = _price_panel_for_tickers(
        canonical,
        start_date,
        end_date,
        price_panel=price_panel,
        cache_dir=cache_dir,
    )
    if prices.empty:
        return pd.DataFrame(columns=list(_GP_OUTPUT_COLS))

    ticker_to_cik = _load_ticker_to_cik(cache_dir, user_agent=ua)
    empty_fund = pd.DataFrame(columns=["date", "gross_profit_ttm", "assets"])

    pieces: list[pd.DataFrame] = []
    for ticker in sorted(prices["ticker"].unique()):
        cik10 = _lookup_cik(ticker, ticker_to_cik)
        if cik10 is None:
            logger.warning("No SEC CIK for ticker %s; leaving GP fundamentals NaN", ticker)
            fund = empty_fund
        else:
            try:
                facts = _load_companyfacts(cik10, cache_dir, user_agent=ua)
                fund = _extract_gp_fundamentals(facts)
            except requests.HTTPError as exc:
                logger.warning(
                    "CompanyFacts fetch failed for %s (CIK %s): %s",
                    ticker,
                    cik10,
                    exc,
                )
                fund = empty_fund

        ticker_prices = prices[prices["ticker"] == ticker]
        pieces.append(_align_gp_to_prices(ticker_prices, fund))

    if not pieces:
        return pd.DataFrame(columns=list(_GP_OUTPUT_COLS))
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def fetch_filing_clock_daily(
    tickers: list[str],
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    *,
    price_panel: pd.DataFrame | None = None,
    forms: tuple[str, ...] = _FILING_FORMS,
    cache_dir: str = DEFAULT_CACHE_DIR,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """
    PIT filing-clock anchors on the trading calendar.

    Columns: ``date``, ``ticker``, ``last_filed``, ``expected_next_filed``.

    Distinct 10-Q/10-K(+/A) filing dates are collected from cached CompanyFacts
    JSON. At each filing event ``i``, ``expected_next_filed`` is forecast using
    only events ``0..i`` (same-slot last year when ``i >= 3``, else median of
    the last up-to-4 gaps). Anchors are held forward onto trading days via
    ``merge_asof(..., direction='backward')``.

    Join onto an OHLCV / S1 panel with::

        panel.merge(clock, on=["date", "ticker"], how="left")

    On the S1 trade-date panel, prefer merging on ``feature_date`` (rename
    clock ``date`` → ``feature_date``) so same-morning filings cannot leak.
    Pass ``price_panel`` when OHLCV is already loaded to avoid re-downloading.
    """
    if not forms:
        raise ValueError("forms must be a non-empty tuple of form strings")
    canonical = _validate_tickers(tickers)
    ua = _resolve_user_agent(user_agent)
    os.makedirs(cache_dir, exist_ok=True)

    prices = _price_panel_for_tickers(
        canonical,
        start_date,
        end_date,
        price_panel=price_panel,
        cache_dir=cache_dir,
    )
    if prices.empty:
        return pd.DataFrame(columns=list(_FILING_CLOCK_OUTPUT_COLS))

    ticker_to_cik = _load_ticker_to_cik(cache_dir, user_agent=ua)
    empty_events = pd.DataFrame(
        columns=["date", "last_filed", "expected_next_filed"]
    )

    pieces: list[pd.DataFrame] = []
    for ticker in sorted(prices["ticker"].unique()):
        cik10 = _lookup_cik(ticker, ticker_to_cik)
        if cik10 is None:
            logger.warning(
                "No SEC CIK for ticker %s; leaving filing-clock anchors NaT",
                ticker,
            )
            events = empty_events
        else:
            try:
                facts = _load_companyfacts(cik10, cache_dir, user_agent=ua)
                events = _extract_filing_clock_events(facts, forms)
            except requests.HTTPError as exc:
                logger.warning(
                    "CompanyFacts fetch failed for %s (CIK %s): %s",
                    ticker,
                    cik10,
                    exc,
                )
                events = empty_events

        ticker_prices = prices[prices["ticker"] == ticker]
        pieces.append(_align_filing_clock_to_prices(ticker_prices, events))

    if not pieces:
        return pd.DataFrame(columns=list(_FILING_CLOCK_OUTPUT_COLS))
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
