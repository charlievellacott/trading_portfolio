"""
FINRA Regulation SHO daily short-sale volume fetcher (H-010 Short-Selling Pressure).

Downloads off-exchange short volume by security from FINRA CDN daily files
(``FNSQ`` / ``FNYX`` / ``FNRA`` by default; excludes ``FNQC`` and ``CNMS``),
caches full-universe year files, and returns a long panel summed across
facilities.

Progress: calendar-month ``tqdm`` on stderr (``FINRA short vol``). Empty /
holiday CDN responses are recorded in ``{facility}_{year}_empty.parquet``
sidecars so they are not re-probed. Missing days within a month download in
parallel via ``ThreadPoolExecutor`` (default ``max_workers=8``).

Symbology
---------
``_canonical_finra_ticker`` keeps the project form (strip + upper), e.g.
``BRK.B``. FINRA daily files typically use a slash for share classes
(``BRK/B``). Matching also tries aliases without the separator and with
``.`` ↔ ``/``. Notebook / research coverage checks should verify rare
symbols.

Factor math does not live here — callers merge onto an OHLCV panel on
``['date', 'ticker']``. Dates are calendar / feature dates (FINRA trade
date of the short-volume file), posted ~18:00 ET same day.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

import pandas as pd
import requests
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

FINRA_DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/{facility}shvol{yyyymmdd}.txt"
DEFAULT_FACILITIES = ("FNSQ", "FNYX", "FNRA")  # exclude FNQC and CNMS
DEFAULT_MAX_WORKERS = 8
_USER_AGENT = (
    "Mozilla/5.0 (compatible; trading_portfolio/0.1; "
    "+https://github.com/local/trading_portfolio)"
)

# alternative_data/ -> ingestion/ -> 01_data/ -> cache/
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "cache",
)

_REQUEST_SLEEP_SEC = 0.15
_CACHE_SUBDIR = "finra_short_volume"

_OUTPUT_COLS = (
    "date",
    "ticker",
    "short_volume",
    "short_exempt_volume",
    "total_volume",
)

_RAW_COLS = (
    "date",
    "ticker",
    "short_volume",
    "short_exempt_volume",
    "total_volume",
)


def _canonical_finra_ticker(symbol: str) -> str:
    """Project ticker form used as the return ``ticker`` key (strip + upper).

    Does not rewrite separators. Matching against FINRA ``Symbol`` values
    uses :func:`_ticker_aliases` (dots / slashes / concatenated). Coverage
    for edge cases should be verified in the research notebook.
    """
    return symbol.strip().upper()


def _ticker_aliases(symbol: str) -> tuple[str, ...]:
    """FINRA Symbol spellings that may map to a project ticker."""
    canon = _canonical_finra_ticker(symbol)
    aliases = [canon]
    if "." in canon:
        aliases.append(canon.replace(".", "/"))
        aliases.append(canon.replace(".", ""))
    if "/" in canon:
        aliases.append(canon.replace("/", "."))
        aliases.append(canon.replace("/", ""))
    if "-" in canon:
        aliases.append(canon.replace("-", "/"))
        aliases.append(canon.replace("-", "."))
        aliases.append(canon.replace("-", ""))
    # Preserve order, drop duplicates.
    return tuple(dict.fromkeys(aliases))


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_OUTPUT_COLS))


def _to_timestamp(value: date | str | pd.Timestamp | datetime) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _resolve_date_range(
    start_date: str | date | None,
    end_date: str | date | None,
    price_panel: pd.DataFrame | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = start_date
    end = end_date
    if price_panel is not None and not price_panel.empty and "date" in price_panel.columns:
        panel_dates = pd.to_datetime(price_panel["date"])
        if start is None:
            start = panel_dates.min()
        if end is None:
            end = panel_dates.max()
    if start is None or end is None:
        raise ValueError(
            "start_date and end_date are required when price_panel cannot "
            "infer the range (pass both, or a non-empty price_panel with a "
            "'date' column)"
        )
    start_ts = _to_timestamp(start)
    end_ts = _to_timestamp(end)
    if end_ts < start_ts:
        raise ValueError(f"end_date {end_ts.date()} is before start_date {start_ts.date()}")
    return start_ts, end_ts


def _iter_month_windows(
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Inclusive calendar-month slices overlapping ``[start_date, end_date]``."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        return []
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(year=start.year, month=start.month, day=1)
    while cursor <= end:
        m_start = max(start, cursor)
        next_month = cursor + pd.offsets.MonthBegin(1)
        m_end = min(end, next_month - pd.Timedelta(days=1))
        windows.append((m_start, m_end))
        cursor = next_month
    return windows


def _year_cache_path(cache_dir: str, facility: str, year: int) -> str:
    return os.path.join(cache_dir, _CACHE_SUBDIR, f"{facility}_{year}.parquet")


def _empty_dates_path(cache_dir: str, facility: str, year: int) -> str:
    return os.path.join(
        cache_dir, _CACHE_SUBDIR, f"{facility}_{year}_empty.parquet"
    )


def parse_finra_short_volume_text(text: str) -> pd.DataFrame:
    """Parse a FINRA daily Reg SHO short-volume file body.

    Drops the header, trailer / record-count lines, blank symbols, and rows
    whose numeric fields fail coercion.
    """
    if not text or not text.strip():
        return _empty_frame()

    rows: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Header
        if line.lower().startswith("date|"):
            continue
        parts = line.split("|")
        # Trailer is typically a bare record count (no pipes).
        if len(parts) < 5:
            continue
        symbol = parts[1].strip()
        if not symbol:
            continue
        date_raw = parts[0].strip()
        try:
            dt = pd.to_datetime(date_raw, format="%Y%m%d")
        except (ValueError, TypeError):
            continue
        try:
            short_vol = float(parts[2].strip())
            # Layout may omit ShortExemptVolume on very old files (5 fields).
            if len(parts) >= 6:
                short_exempt = float(parts[3].strip())
                total_vol = float(parts[4].strip())
            else:
                short_exempt = 0.0
                total_vol = float(parts[3].strip())
        except (ValueError, TypeError):
            continue
        rows.append(
            {
                "date": dt.normalize(),
                "ticker": symbol.upper(),
                "short_volume": short_vol,
                "short_exempt_volume": short_exempt,
                "total_volume": total_vol,
            }
        )

    if not rows:
        return _empty_frame()
    out = pd.DataFrame(rows)
    return out[list(_RAW_COLS)].reset_index(drop=True)


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


def _download_facility_day(
    facility: str,
    day: pd.Timestamp,
    *,
    session: requests.Session | None = None,
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    """Fetch one facility/day file. 404/403/errors → empty frame (no retries)."""
    yyyymmdd = day.strftime("%Y%m%d")
    url = FINRA_DAILY_URL.format(facility=facility, yyyymmdd=yyyymmdd)
    getter = session.get if session is not None else requests.get
    try:
        resp = getter(url, timeout=60)
    except requests.RequestException as exc:
        logger.warning("FINRA request failed %s %s: %s", facility, yyyymmdd, exc)
        return _empty_frame()

    if resp.status_code == 404:
        logger.debug("FINRA 404 (empty day): %s %s", facility, yyyymmdd)
        return _empty_frame()

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning("FINRA HTTP error %s %s: %s", facility, yyyymmdd, exc)
        return _empty_frame()

    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return parse_finra_short_volume_text(resp.text)


def _load_year_cache(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return _empty_frame()
    try:
        cached = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 — corrupt cache → refetch year
        logger.warning("FINRA cache unreadable %s: %s", path, exc)
        return _empty_frame()
    if cached.empty:
        return _empty_frame()
    cached = cached.copy()
    cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
    cached["ticker"] = cached["ticker"].astype(str).str.upper()
    for col in ("short_volume", "short_exempt_volume", "total_volume"):
        if col not in cached.columns:
            cached[col] = 0.0
        cached[col] = pd.to_numeric(cached[col], errors="coerce")
    cached = cached.dropna(subset=["ticker", "short_volume", "total_volume"])
    cached = cached[cached["ticker"].str.len() > 0]
    return cached[list(_RAW_COLS)].reset_index(drop=True)


def _write_year_cache(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    if out.empty:
        out = _empty_frame()
    else:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out = out[list(_RAW_COLS)]
    out.to_parquet(path, index=False)


def _load_empty_dates(path: str) -> set[date]:
    if not os.path.isfile(path):
        return set()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FINRA empty-dates cache unreadable %s: %s", path, exc)
        return set()
    if df.empty or "date" not in df.columns:
        return set()
    return set(pd.to_datetime(df["date"]).dt.normalize().dt.date)


def _write_empty_dates(path: str, empty_dates: set[date]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not empty_dates:
        # Keep an empty parquet so the path exists and load returns empty set.
        pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")}).to_parquet(
            path, index=False
        )
        return
    dates = sorted(empty_dates)
    out = pd.DataFrame({"date": pd.to_datetime(dates)})
    out.to_parquet(path, index=False)


def _merge_year_pieces(
    cached: pd.DataFrame, new_pieces: list[pd.DataFrame]
) -> pd.DataFrame:
    pieces = [cached] if not cached.empty else []
    pieces.extend(p for p in new_pieces if p is not None and not p.empty)
    if not pieces:
        return _empty_frame()
    merged = pd.concat(pieces, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.drop_duplicates(subset=["date", "ticker"], keep="last")
    return merged.sort_values(["date", "ticker"], kind="mergesort").reset_index(
        drop=True
    )


def _download_missing_days(
    facility: str,
    missing: list[pd.Timestamp],
    *,
    session: requests.Session,
    max_workers: int,
) -> tuple[list[pd.DataFrame], list[date]]:
    """Download ``missing`` days; return (non-empty frames, empty calendar dates)."""
    if not missing:
        return [], []

    workers = max(1, int(max_workers))
    sleep_sec = _REQUEST_SLEEP_SEC if workers == 1 else 0.0
    data_frames: list[pd.DataFrame] = []
    empty_days: list[date] = []

    def _one(day: pd.Timestamp) -> tuple[pd.Timestamp, pd.DataFrame]:
        return day, _download_facility_day(
            facility, day, session=session, sleep_sec=sleep_sec
        )

    if workers == 1:
        for day in missing:
            d, frame = _one(day)
            if frame.empty:
                empty_days.append(d.date())
            else:
                data_frames.append(frame)
        return data_frames, empty_days

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, day) for day in missing]
        for fut in as_completed(futures):
            d, frame = fut.result()
            if frame.empty:
                empty_days.append(d.date())
            else:
                data_frames.append(frame)
    return data_frames, empty_days


def _ensure_facility_month(
    facility: str,
    month_days: list[pd.Timestamp],
    cache_dir: str,
    *,
    session: requests.Session,
    max_workers: int,
) -> pd.DataFrame:
    """
    Ensure year cache covers ``month_days`` for ``facility``.

    Downloads only missing (not in data cache and not in empty sidecar) days.
    Returns the year cache frame (may include dates outside the month).
    """
    if not month_days:
        return _empty_frame()

    year = int(month_days[0].year)
    path = _year_cache_path(cache_dir, facility, year)
    empty_path = _empty_dates_path(cache_dir, facility, year)
    cached = _load_year_cache(path)
    empty_dates = _load_empty_dates(empty_path)

    have: set[date] = set(empty_dates)
    if not cached.empty:
        have |= set(pd.to_datetime(cached["date"]).dt.normalize().dt.date)

    missing = [d for d in month_days if d.date() not in have]
    if not missing:
        return cached

    logger.info(
        "FINRA %s %d: downloading %d missing day(s)",
        facility,
        year,
        len(missing),
    )
    new_frames, new_empty = _download_missing_days(
        facility, missing, session=session, max_workers=max_workers
    )
    if new_empty:
        empty_dates |= set(new_empty)
        _write_empty_dates(empty_path, empty_dates)

    merged = _merge_year_pieces(cached, new_frames)
    _write_year_cache(path, merged)
    return merged


def _build_alias_lookup(tickers: list[str]) -> dict[str, str]:
    """Map every FINRA alias → project canonical ticker."""
    lookup: dict[str, str] = {}
    for raw in tickers:
        canon = _canonical_finra_ticker(raw)
        if not canon:
            continue
        for alias in _ticker_aliases(canon):
            lookup.setdefault(alias, canon)
    return lookup


def _filter_to_tickers(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if df.empty or not tickers:
        return _empty_frame()
    lookup = _build_alias_lookup(tickers)
    if not lookup:
        return _empty_frame()
    mapped = df.copy()
    mapped["ticker"] = mapped["ticker"].map(lookup)
    mapped = mapped.dropna(subset=["ticker"])
    if mapped.empty:
        return _empty_frame()
    # Same project ticker may match multiple FINRA spellings on one day —
    # sum those rows (should be rare).
    mapped = (
        mapped.groupby(["date", "ticker"], as_index=False)[
            ["short_volume", "short_exempt_volume", "total_volume"]
        ]
        .sum()
    )
    return mapped[list(_OUTPUT_COLS)].reset_index(drop=True)


def fetch_short_volume_daily(
    tickers: list[str],
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    *,
    facilities: tuple[str, ...] = DEFAULT_FACILITIES,
    price_panel: pd.DataFrame | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> pd.DataFrame:
    """Daily off-exchange short-sale volume summed across FINRA facilities.

    Columns: ``date``, ``ticker``, ``short_volume``, ``short_exempt_volume``,
    ``total_volume``. Dates are calendar / feature dates (trade date of the
    short volume file).

    Parameters
    ----------
    tickers
        Project tickers (e.g. ``BRK.B``). Matching tries FINRA aliases
        (``BRK/B``, ``BRKB``, …).
    start_date, end_date
        Inclusive range. If omitted, inferred from ``price_panel['date']``.
    facilities
        FINRA file prefixes. Default ``FNSQ``, ``FNYX``, ``FNRA``.
    price_panel
        Optional OHLCV panel used only to infer the date range.
    cache_dir
        Root cache directory; year files live under
        ``{cache_dir}/finra_short_volume/{FACILITY}_{YYYY}.parquet``.
        Probed-empty days: ``{FACILITY}_{YYYY}_empty.parquet``.
    max_workers
        Parallel CDN downloads per facility-month (default 8). Use ``1`` for
        sequential polite mode (includes short sleep after successes).
    """
    if not tickers:
        return _empty_frame()
    if not facilities:
        return _empty_frame()

    start_ts, end_ts = _resolve_date_range(start_date, end_date, price_panel)
    # Business days only — weekends never requested; holidays recorded empty.
    needed = list(pd.bdate_range(start_ts, end_ts))
    if not needed:
        return _empty_frame()

    month_windows = _iter_month_windows(start_ts, end_ts)
    fac_list = [f.strip().upper() for f in facilities if f and str(f).strip()]
    if not fac_list or not month_windows:
        return _empty_frame()

    session = _make_session()
    # facility → year → year frame (updated as months complete)
    year_frames: dict[str, dict[int, pd.DataFrame]] = {f: {} for f in fac_list}

    for m_start, m_end in tqdm(
        month_windows,
        desc="FINRA short vol",
        unit="month",
        file=sys.stderr,
    ):
        month_days = [d for d in needed if m_start <= d <= m_end]
        if not month_days:
            continue
        for fac in fac_list:
            year_df = _ensure_facility_month(
                fac,
                month_days,
                cache_dir,
                session=session,
                max_workers=max_workers,
            )
            year_frames[fac][int(m_start.year)] = year_df

    facility_frames: list[pd.DataFrame] = []
    for fac, by_year in year_frames.items():
        for _year, year_df in by_year.items():
            if year_df is None or year_df.empty:
                continue
            mask = (year_df["date"] >= start_ts) & (year_df["date"] <= end_ts)
            sliced = year_df.loc[mask]
            if not sliced.empty:
                facility_frames.append(sliced)

    if not facility_frames:
        return _empty_frame()

    combined = pd.concat(facility_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined["ticker"] = combined["ticker"].astype(str).str.upper()

    summed = (
        combined.groupby(["date", "ticker"], as_index=False)[
            ["short_volume", "short_exempt_volume", "total_volume"]
        ]
        .sum()
    )
    out = _filter_to_tickers(summed, tickers)
    if out.empty:
        return _empty_frame()
    out = out.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    return out[list(_OUTPUT_COLS)]
