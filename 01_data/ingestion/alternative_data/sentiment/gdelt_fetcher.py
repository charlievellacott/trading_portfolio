"""
GDELT 2.0 GKG sentiment fetcher (H-009).

History: BigQuery ``gdelt-bq.gdeltv2.gkg_partitioned`` (ADC; always filter
``_PARTITIONTIME``). Match + daily median/count run in SQL over calendar-month
windows (one partition scan per month). Live: 15-minute ``*.gkg.csv.zip`` via
``lastupdate.txt``.

Entity resolution: SEC ``company_tickers.json`` titles → auto aliases, plus
hand-editable ``gdelt_company_name_map.csv`` (``company_name,ticker``) beside
this module. Substring match on GKG ``V2Organizations``.

Console logging: **only** ``NOT_FOUND`` / ``AMBIGUOUS`` lines
(``GDELT_MAP\\tSTATUS\\t<company name>``). OK tickers are silent.
Operational progress (tqdm) goes to stderr.

Factor math lives in ``feature_implementation.gdelt_sentiment``. Prefer
``add_gdelt_sentiment_factors(..., sentiment_data_exists=False)`` which
fetches and merges on S1 ``feature_date`` (or ``date``) internally; pass
``sentiment_data_exists=True`` when ``median_tone`` / ``n_articles`` are
already on the panel.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import sys
import zipfile
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm

from data.ingestion.equity_fetcher import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

_HERE = os.path.abspath(__file__)
_SENTIMENT_DIR = os.path.dirname(_HERE)
DEFAULT_COMPANY_NAME_MAP = os.path.join(_SENTIMENT_DIR, "gdelt_company_name_map.csv")

GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
BQ_GKG_TABLE = "`gdelt-bq.gdeltv2.gkg_partitioned`"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# First day of GDELT 2.0 GKG; earlier partitions are empty for gdeltv2.
GDELT_V2_START = "2015-02-18"
# BQ / query aliases shorter than this are dropped (too many false hits).
_MIN_QUERY_ALIAS_LEN = 4
_SCORED_COLS = ("date", "ticker", "tone")

GKG_COL_DATE = 1
GKG_COL_V2ORGS = 14
GKG_COL_V2TONE = 15

_LEGAL_SUFFIXES = (
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "ltd",
    "ltd.",
    "llc",
    "plc",
    "nv",
    "sa",
    "ag",
    "company",
)
_FIRST_TOKEN_BLOCKLIST = frozenset(
    {"the", "and", "bank", "group", "holdings", "holding", "international"}
)
# Extra generic tokens blocked for BigQuery LIKE/STRPOS matching only.
_QUERY_ALIAS_BLOCKLIST = _FIRST_TOKEN_BLOCKLIST | frozenset(
    {
        "corp",
        "corporation",
        "company",
        "inc",
        "ltd",
        "llc",
        "plc",
        "technologies",
        "technology",
        "systems",
        "solutions",
        "services",
        "partners",
        "capital",
        "financial",
        "global",
        "united",
        "american",
        "national",
    }
)

_OUTPUT_COLS = ("date", "ticker", "median_tone", "n_articles")


# ---------------------------------------------------------------------------
# Tone / alias helpers
# ---------------------------------------------------------------------------


def parse_v2tone(raw: Any) -> float | None:
    """First field of V2Tone CSV string → overall document tone, else None."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    first = s.split(",")[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


def build_aliases(title: str, extra: Iterable[str] | None = None) -> list[str]:
    """Build case-insensitive org aliases from an SEC title (+ optional extras)."""
    aliases: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = " ".join(str(s).split())
        if not s:
            return
        key = s.casefold()
        if key in seen:
            return
        seen.add(key)
        aliases.append(s)

    title = " ".join(str(title).split())
    if title:
        _add(title)
        parts = title.split()
        while parts and parts[-1].rstrip(".").casefold() in _LEGAL_SUFFIXES:
            parts = parts[:-1]
        stripped = " ".join(parts)
        if stripped:
            _add(stripped)
            tok = parts[0] if parts else ""
            if len(tok) >= 3 and tok.casefold() not in _FIRST_TOKEN_BLOCKLIST:
                _add(tok)

    if extra:
        for e in extra:
            _add(e)
    return aliases


def load_company_name_map(path: str | None = None) -> dict[str, list[str]]:
    """Load ``company_name,ticker`` CSV → ticker → list of company names."""
    path = path or DEFAULT_COMPANY_NAME_MAP
    out: dict[str, list[str]] = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("company_name") or "").strip()
            ticker = (row.get("ticker") or "").strip().upper()
            if not name or not ticker:
                continue
            out.setdefault(ticker, [])
            if name not in out[ticker]:
                out[ticker].append(name)
    return out


def _sec_ticker_map(cache_dir: str) -> dict[str, dict[str, str]]:
    """SEC company_tickers.json → ticker → {cik, title} (project ticker form)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "sec_company_tickers.json")
    if not os.path.isfile(path):
        resp = requests.get(
            SEC_TICKERS_URL,
            headers={"User-Agent": "trading_portfolio research bot"},
            timeout=60,
        )
        resp.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)
    import json

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict[str, str]] = {}
    for row in raw.values():
        sec_ticker = str(row.get("ticker", "")).strip().upper()
        if not sec_ticker:
            continue
        project_ticker = sec_ticker.replace("-", ".")
        out[project_ticker] = {
            "cik": str(row.get("cik_str", "")).zfill(10),
            "title": str(row.get("title", "")).strip(),
        }
    return out


def build_gdelt_alias_table(
    tickers: Sequence[str],
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    company_name_map_path: str | None = None,
) -> pd.DataFrame:
    """
    Build alias rows for ``tickers``.

    Columns: ``ticker``, ``cik``, ``title``, ``n_aliases``, ``aliases`` (pipe-joined).
    """
    gdelt_cache = os.path.join(cache_dir, "alternative_data", "gdelt")
    os.makedirs(gdelt_cache, exist_ok=True)
    sec = _sec_ticker_map(gdelt_cache)
    csv_map = load_company_name_map(company_name_map_path)
    rows = []
    for t in sorted({str(x).strip().upper() for x in tickers}):
        info = sec.get(t, {})
        title = info.get("title", "")
        extras = csv_map.get(t, [])
        aliases = build_aliases(title, extra=extras)
        rows.append(
            {
                "ticker": t,
                "cik": info.get("cik", ""),
                "title": title,
                "n_aliases": len(aliases),
                "aliases": " | ".join(aliases),
            }
        )
    return pd.DataFrame(rows)


def _alias_map_from_table(alias_df: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for _, row in alias_df.iterrows():
        aliases = [a.strip() for a in str(row["aliases"]).split("|") if a.strip()]
        out[str(row["ticker"]).upper()] = aliases
    return out


def match_orgs_to_ticker(v2orgs: str, alias_map: dict[str, list[str]]) -> list[str]:
    """Return tickers whose aliases appear as substrings in V2Organizations."""
    if not v2orgs or (isinstance(v2orgs, float) and pd.isna(v2orgs)):
        return []
    text = str(v2orgs).casefold()
    hits: list[str] = []
    for ticker, aliases in alias_map.items():
        for alias in aliases:
            a = alias.casefold()
            if len(a) >= 2 and a in text:
                hits.append(ticker)
                break
    return hits


def _log_map_issue(status: str, company_name: str) -> None:
    """Print a copy-paste-friendly mapping problem line (problems only)."""
    print(f"GDELT_MAP\t{status}\t{company_name}", flush=True)


def _detect_ambiguous_aliases(alias_map: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Return (ticker, alias) pairs whose casefold alias is shared by another ticker."""
    inv: dict[str, list[str]] = {}
    for ticker, aliases in alias_map.items():
        for a in aliases:
            inv.setdefault(a.casefold(), []).append(ticker)
    issues: list[tuple[str, str]] = []
    for a_cf, tickers in inv.items():
        uniq = sorted(set(tickers))
        if len(uniq) < 2:
            continue
        # recover a display name
        display = a_cf
        for t in uniq:
            for a in alias_map[t]:
                if a.casefold() == a_cf:
                    display = a
                    break
        for t in uniq:
            issues.append((t, display))
    return issues


def clamp_gdelt_v2_start(start_date: str | date | pd.Timestamp) -> str:
    """Return ``max(start_date, GDELT_V2_START)`` as ``YYYY-MM-DD``."""
    start = pd.Timestamp(start_date).normalize()
    floor = pd.Timestamp(GDELT_V2_START)
    return max(start, floor).strftime("%Y-%m-%d")


def iter_year_windows(
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> list[tuple[str, str]]:
    """Inclusive calendar-year slices overlapping ``[start_date, end_date]``."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        return []
    windows: list[tuple[str, str]] = []
    year = start.year
    while year <= end.year:
        y_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        y_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        windows.append(
            (y_start.strftime("%Y-%m-%d"), y_end.strftime("%Y-%m-%d"))
        )
        year += 1
    return windows


def iter_month_windows(
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> list[tuple[str, str]]:
    """Inclusive calendar-month slices overlapping ``[start_date, end_date]``."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        return []
    windows: list[tuple[str, str]] = []
    cursor = pd.Timestamp(year=start.year, month=start.month, day=1)
    while cursor <= end:
        m_start = max(start, cursor)
        next_month = cursor + pd.offsets.MonthBegin(1)
        m_end = min(end, next_month - pd.Timedelta(days=1))
        windows.append(
            (m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))
        )
        cursor = next_month
    return windows


def _alias_fingerprint(aliases: Sequence[str]) -> str:
    payload = "\n".join(sorted(a.casefold() for a in aliases)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def is_query_alias(alias: str, *, min_len: int = _MIN_QUERY_ALIAS_LEN) -> bool:
    """True if ``alias`` is long / specific enough for BigQuery org matching."""
    a = " ".join(str(alias).split())
    if len(a) < min_len:
        return False
    if a.casefold() in _QUERY_ALIAS_BLOCKLIST:
        return False
    return True


def filter_alias_map_for_query(
    alias_map: dict[str, list[str]],
    *,
    min_len: int = _MIN_QUERY_ALIAS_LEN,
) -> dict[str, list[str]]:
    """Drop short / generic aliases from ``alias_map`` (BQ query hygiene)."""
    out: dict[str, list[str]] = {}
    for ticker, aliases in alias_map.items():
        kept = [a for a in aliases if is_query_alias(a, min_len=min_len)]
        if kept:
            out[ticker] = kept
    return out


def scored_chunk_path(
    cache_dir: str,
    *,
    window_start: str,
    window_end: str,
    aliases: Sequence[str],
) -> str:
    """Stable parquet path for one month-window daily panel (all query aliases)."""
    scored_dir = os.path.join(cache_dir, "alternative_data", "gdelt", "bq_scored")
    os.makedirs(scored_dir, exist_ok=True)
    fp = _alias_fingerprint(aliases)
    name = f"{window_start}_{window_end}_{fp}.parquet"
    return os.path.join(scored_dir, name)


# ---------------------------------------------------------------------------
# Aggregation / scoring
# ---------------------------------------------------------------------------


def _aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """``date, ticker, tone`` → ``date, ticker, median_tone, n_articles``."""
    if scored.empty:
        return pd.DataFrame(columns=list(_OUTPUT_COLS))
    g = (
        scored.groupby(["date", "ticker"], sort=False)["tone"]
        .agg(median_tone="median", n_articles="count")
        .reset_index()
    )
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    g["ticker"] = g["ticker"].astype(str).str.upper()
    return g[list(_OUTPUT_COLS)]


def _combine_daily_panels(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge daily panels; overlapping keys use n_articles-weighted mean tone."""
    nonempty = [p for p in parts if p is not None and not p.empty]
    if not nonempty:
        return pd.DataFrame(columns=list(_OUTPUT_COLS))
    d = pd.concat(nonempty, ignore_index=True)
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["ticker"] = d["ticker"].astype(str).str.upper()
    d["median_tone"] = pd.to_numeric(d["median_tone"], errors="coerce")
    d["n_articles"] = pd.to_numeric(d["n_articles"], errors="coerce").fillna(0)
    d = d.dropna(subset=["median_tone"])
    if d.empty:
        return pd.DataFrame(columns=list(_OUTPUT_COLS))
    d["_w"] = d["median_tone"] * d["n_articles"]
    g = (
        d.groupby(["date", "ticker"], sort=False)
        .agg(_w=("_w", "sum"), n_articles=("n_articles", "sum"))
        .reset_index()
    )
    g["median_tone"] = g["_w"] / g["n_articles"].clip(lower=1)
    g["n_articles"] = g["n_articles"].astype(int)
    return g[list(_OUTPUT_COLS)]


def _score_gkg_frame_ref(
    raw: pd.DataFrame,
    alias_map: dict[str, list[str]],
    *,
    date_col: str,
    orgs_col: str,
    tone_col: str,
) -> pd.DataFrame:
    """Row-wise reference scorer (tests / equivalence only)."""
    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        tone = parse_v2tone(r.get(tone_col))
        if tone is None:
            continue
        date_raw = str(r.get(date_col, ""))
        if len(date_raw) < 8:
            continue
        day = date_raw[:8]
        try:
            d = pd.Timestamp(datetime.strptime(day, "%Y%m%d"))
        except ValueError:
            continue
        tickers = match_orgs_to_ticker(r.get(orgs_col, ""), alias_map)
        for t in tickers:
            rows.append({"date": d, "ticker": t, "tone": tone})
    if not rows:
        return pd.DataFrame(columns=list(_SCORED_COLS))
    return pd.DataFrame(rows)


def _score_gkg_frame(
    raw: pd.DataFrame,
    alias_map: dict[str, list[str]],
    *,
    date_col: str,
    orgs_col: str,
    tone_col: str,
) -> pd.DataFrame:
    """Vectorized GKG row → ``date, ticker, tone`` (same semantics as ref)."""
    if raw.empty:
        return pd.DataFrame(columns=list(_SCORED_COLS))

    tones = raw[tone_col].map(parse_v2tone)
    days = raw[date_col].astype(str).str.slice(0, 8)
    dates = pd.to_datetime(days, format="%Y%m%d", errors="coerce")
    valid = tones.notna() & dates.notna()
    if not valid.any():
        return pd.DataFrame(columns=list(_SCORED_COLS))

    orgs = raw[orgs_col].fillna("").astype(str).str.casefold()
    parts: list[pd.DataFrame] = []
    for ticker, aliases in alias_map.items():
        mask = pd.Series(False, index=raw.index)
        for alias in aliases:
            a = alias.casefold()
            if len(a) < 2:
                continue
            mask = mask | orgs.str.contains(a, regex=False, na=False)
        hit = valid & mask
        if not hit.any():
            continue
        parts.append(
            pd.DataFrame(
                {
                    "date": dates.loc[hit].to_numpy(),
                    "ticker": ticker,
                    "tone": tones.loc[hit].astype(float).to_numpy(),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=list(_SCORED_COLS))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# BigQuery history + live HTTP
# ---------------------------------------------------------------------------


def _bq_client(project: str | None = None):
    from google.cloud import bigquery

    proj = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "trading-repository")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", proj)
    return bigquery.Client(project=proj)


def _empty_scored() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_SCORED_COLS))


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_OUTPUT_COLS))


def _normalize_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_daily()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["median_tone"] = pd.to_numeric(out["median_tone"], errors="coerce")
    out["n_articles"] = (
        pd.to_numeric(out["n_articles"], errors="coerce").fillna(0).astype(int)
    )
    out = out.dropna(subset=["median_tone"])
    if out.empty:
        return _empty_daily()
    return out[list(_OUTPUT_COLS)]


def _read_daily_chunk(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    df = pd.read_parquet(path)
    return _normalize_daily_frame(df)


def _write_daily_chunk(path: str, daily: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = daily if not daily.empty else _empty_daily()
    out.to_parquet(path, index=False)


def _job_to_dataframe(job_result: Any) -> pd.DataFrame:
    """Download query results; prefer BigQuery Storage API, fall back to REST."""
    try:
        return job_result.to_dataframe(create_bqstorage_client=True)
    except Exception as exc:  # noqa: BLE001 — storage client optional
        logger.info("BQ Storage API unavailable (%s); falling back to REST", exc)
        return job_result.to_dataframe(create_bqstorage_client=False)


def _sql_string_literal(value: str) -> str:
    """Escape a Python string for a BigQuery single-quoted string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _alias_structs_sql(alias_map: dict[str, list[str]]) -> str:
    """Build ``STRUCT('alias' AS alias, 'TICKER' AS ticker), ...`` list."""
    rows: list[str] = []
    seen: set[tuple[str, str]] = set()
    for ticker, aliases in sorted(alias_map.items()):
        t = str(ticker).strip().upper()
        for alias in aliases:
            a = " ".join(str(alias).split()).casefold()
            if not a:
                continue
            key = (a, t)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                f"STRUCT({_sql_string_literal(a)} AS alias, "
                f"{_sql_string_literal(t)} AS ticker)"
            )
    return ",\n    ".join(rows)


def _query_gkg_daily(
    client: Any,
    alias_map: dict[str, list[str]],
    window_start: str,
    window_end: str,
) -> pd.DataFrame:
    """
    One month-window scan: match all aliases in SQL, return daily median + count.

    Returns ``date, ticker, median_tone, n_articles``.
    """
    structs = _alias_structs_sql(alias_map)
    if not structs:
        return _empty_daily()

    sql = f"""
    WITH aliases AS (
      SELECT * FROM UNNEST([
        {structs}
      ])
    ),
    gkg AS (
      SELECT
        SAFE.PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS d,
        LOWER(V2Organizations) AS orgs,
        SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone
      FROM {BQ_GKG_TABLE}
      WHERE DATE(_PARTITIONTIME) BETWEEN DATE('{window_start}') AND DATE('{window_end}')
        AND V2Organizations IS NOT NULL
        AND V2Tone IS NOT NULL
    ),
    matched AS (
      SELECT g.d AS date, a.ticker, g.tone
      FROM gkg g
      INNER JOIN aliases a
        ON STRPOS(g.orgs, a.alias) > 0
      WHERE g.d IS NOT NULL AND g.tone IS NOT NULL
    )
    SELECT
      date,
      ticker,
      APPROX_QUANTILES(tone, 100)[OFFSET(50)] AS median_tone,
      COUNT(*) AS n_articles
    FROM matched
    GROUP BY date, ticker
    """
    job = client.query(sql)
    raw = _job_to_dataframe(job.result())
    if raw.empty:
        return _empty_daily()
    return _normalize_daily_frame(raw)


def _fetch_history_bq(
    alias_map: dict[str, list[str]],
    start_date: str,
    end_date: str,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    resume: bool = True,
    project: str | None = None,
) -> pd.DataFrame:
    """
    Pull GDELT history via BigQuery; return daily ``median_tone`` / ``n_articles``.

    One query per calendar-month window; results cached under
    ``cache_dir/alternative_data/gdelt/bq_scored/``.
    """
    query_map = filter_alias_map_for_query(alias_map)
    aliases = sorted({a for als in query_map.values() for a in als})
    if not aliases:
        return _empty_daily()

    effective_start = clamp_gdelt_v2_start(start_date)
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts < pd.Timestamp(effective_start):
        return _empty_daily()

    month_windows = iter_month_windows(effective_start, end_date)
    if not month_windows:
        return _empty_daily()

    client: Any | None = None
    parts: list[pd.DataFrame] = []
    for w_start, w_end in tqdm(
        month_windows,
        desc="GDELT BQ",
        unit="month",
        file=sys.stderr,
    ):
        path = scored_chunk_path(
            cache_dir,
            window_start=w_start,
            window_end=w_end,
            aliases=aliases,
        )
        if resume:
            cached = _read_daily_chunk(path)
            if cached is not None:
                if not cached.empty:
                    parts.append(cached)
                continue

        if client is None:
            client = _bq_client(project)
        daily = _query_gkg_daily(client, query_map, w_start, w_end)
        _write_daily_chunk(path, daily)
        if not daily.empty:
            parts.append(daily)

    if not parts:
        return _empty_daily()
    return _combine_daily_panels(parts)


def _gkg_urls_from_lastupdate(n_files: int = 4) -> list[str]:
    text = requests.get(GDELT_LASTUPDATE_URL, timeout=30).text
    stamps: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        url = parts[-1] if parts else ""
        if "gkg.csv.zip" in url:
            # http://.../YYYYMMDDHHMMSS.gkg.csv.zip
            base = os.path.basename(url)
            stamp = base.split(".")[0]
            stamps.append(stamp)
            break
    if not stamps:
        return []
    latest = stamps[0]
    dt = datetime.strptime(latest, "%Y%m%d%H%M%S")
    urls = []
    for i in range(n_files):
        t = dt - pd.Timedelta(minutes=15 * i)
        s = t.strftime("%Y%m%d%H%M%S")
        urls.append(f"http://data.gdeltproject.org/gdeltv2/{s}.gkg.csv.zip")
    return urls


def _fetch_live_http(
    alias_map: dict[str, list[str]],
    *,
    cache_dir: str,
    n_files: int = 96,
) -> pd.DataFrame:
    """Pull recent GKG zips (default ~24h of 15-min files) and score."""
    zip_dir = os.path.join(cache_dir, "alternative_data", "gdelt", "gkg_zips")
    os.makedirs(zip_dir, exist_ok=True)
    urls = _gkg_urls_from_lastupdate(n_files=n_files)
    frames: list[pd.DataFrame] = []
    for url in tqdm(
        urls,
        desc="GDELT live zips",
        unit="file",
        file=sys.stderr,
        disable=len(urls) == 0,
    ):
        name = os.path.basename(url)
        path = os.path.join(zip_dir, name)
        if not os.path.isfile(path):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code != 200:
                    continue
                with open(path, "wb") as f:
                    f.write(r.content)
            except requests.RequestException:
                continue
        try:
            with zipfile.ZipFile(path) as zf:
                member = zf.namelist()[0]
                with zf.open(member) as raw:
                    # GKG has no header; tab-separated
                    data = raw.read().decode("utf-8", errors="replace")
            df = pd.read_csv(
                io.StringIO(data),
                sep="\t",
                header=None,
                usecols=[GKG_COL_DATE, GKG_COL_V2ORGS, GKG_COL_V2TONE],
                names=["gkg_date", "V2Organizations", "V2Tone"],
                dtype=str,
            )
            scored = _score_gkg_frame(
                df,
                alias_map,
                date_col="gkg_date",
                orgs_col="V2Organizations",
                tone_col="V2Tone",
            )
            if not scored.empty:
                frames.append(scored)
        except (OSError, zipfile.BadZipFile, pd.errors.EmptyDataError, ValueError):
            continue
    if not frames:
        return _empty_scored()
    return pd.concat(frames, ignore_index=True)


def _report_mapping_issues(
    tickers: Sequence[str],
    alias_df: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    """Print NOT_FOUND / AMBIGUOUS only (OK tickers silent)."""
    alias_map = _alias_map_from_table(alias_df)
    hit_counts = (
        daily.groupby("ticker")["n_articles"].sum()
        if not daily.empty
        else pd.Series(dtype=float)
    )
    ambiguous = _detect_ambiguous_aliases(alias_map)
    amb_by_ticker: dict[str, list[str]] = {}
    for t, name in ambiguous:
        amb_by_ticker.setdefault(t, [])
        if name not in amb_by_ticker[t]:
            amb_by_ticker[t].append(name)

    printed: set[tuple[str, str]] = set()
    for t in sorted({str(x).strip().upper() for x in tickers}):
        row = alias_df.loc[alias_df["ticker"] == t]
        title = str(row["title"].iloc[0]) if len(row) else ""
        aliases = alias_map.get(t, [])

        if t in amb_by_ticker:
            for name in amb_by_ticker[t]:
                key = ("AMBIGUOUS", name)
                if key not in printed:
                    _log_map_issue("AMBIGUOUS", name)
                    printed.add(key)

        n = float(hit_counts.get(t, 0) or 0)
        if n <= 0:
            # Prefer a CSV/SEC company name for paste-into-CSV workflow
            display = title or (aliases[0] if aliases else t)
            key = ("NOT_FOUND", display)
            if key not in printed:
                _log_map_issue("NOT_FOUND", display)
                printed.add(key)


def fetch_gdelt_sentiment_daily(
    tickers: list[str],
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    *,
    price_panel: pd.DataFrame | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    company_name_map_path: str | None = None,
    use_bigquery: bool = True,
    live_n_files: int = 0,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Return long daily GDELT sentiment: ``date, ticker, median_tone, n_articles``.

    Parameters
    ----------
    tickers:
        Project tickers to cover.
    start_date, end_date:
        Inclusive calendar range for BigQuery history. Inferred from
        ``price_panel`` when omitted. BQ scans clamp to ``GDELT_V2_START``.
    company_name_map_path:
        CSV with ``company_name,ticker`` (default: beside this module).
    use_bigquery:
        If True, pull history from BigQuery GKG (month windows; SQL aggregate).
    live_n_files:
        If > 0, also pull this many recent 15-min GKG zips and merge.
    resume:
        If True, reuse scored month-window daily parquet chunks under
        ``cache_dir/alternative_data/gdelt/bq_scored/``.
    """
    tickers_u = sorted({str(t).strip().upper() for t in tickers})
    if price_panel is not None and not price_panel.empty:
        if start_date is None:
            start_date = pd.to_datetime(price_panel["date"]).min().date()
        if end_date is None:
            end_date = pd.to_datetime(price_panel["date"]).max().date()
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required (or pass price_panel)")

    start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")

    alias_df = build_gdelt_alias_table(
        tickers_u,
        cache_dir=cache_dir,
        company_name_map_path=company_name_map_path,
    )
    alias_map = _alias_map_from_table(alias_df)

    daily_parts: list[pd.DataFrame] = []
    if use_bigquery:
        daily_parts.append(
            _fetch_history_bq(
                alias_map,
                start_s,
                end_s,
                cache_dir=cache_dir,
                resume=resume,
            )
        )
    if live_n_files > 0:
        scored_live = _fetch_live_http(
            alias_map, cache_dir=cache_dir, n_files=live_n_files
        )
        daily_parts.append(_aggregate_daily(scored_live))

    daily = _combine_daily_panels(daily_parts)
    if not daily.empty:
        daily = daily[
            (daily["date"] >= pd.Timestamp(start_s))
            & (daily["date"] <= pd.Timestamp(end_s))
            & (daily["ticker"].isin(tickers_u))
        ].reset_index(drop=True)

    # Cache daily panel
    out_dir = os.path.join(cache_dir, "alternative_data", "gdelt")
    os.makedirs(out_dir, exist_ok=True)
    daily.to_parquet(os.path.join(out_dir, "gdelt_sentiment_daily.parquet"), index=False)
    alias_df.to_parquet(os.path.join(out_dir, "ticker_org_aliases.parquet"), index=False)

    _report_mapping_issues(tickers_u, alias_df, daily)
    return daily.reset_index(drop=True)
