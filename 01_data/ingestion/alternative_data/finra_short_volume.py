"""
FINRA Regulation SHO daily short-sale volume fetcher (H-010 Short-Selling Pressure).

Downloads off-exchange short volume by security from FINRA CDN daily files
(``FNSQ`` / ``FNYX`` / ``FNRA`` by default; excludes ``FNQC`` and ``CNMS``),
caches full-universe year files, and returns a long panel summed across
facilities.

Progress: calendar-month ``tqdm`` on stderr (``FINRA short vol``). Hard-empty
(404 / holiday) CDN responses are recorded in
``{facility}_{year}_empty.parquet`` sidecars so they are not re-probed; soft
failures (403 / network) stay retryable. Completed calendar months are marked
in ``{facility}_{year}_covered.parquet`` so warm runs skip without reloading
year bodies. Missing ``(facility, day)`` jobs within a month download in one
``ThreadPoolExecutor`` (default ``max_workers=8``) with thread-local Sessions.
Year caches load once per facility-year in memory and write once per call.

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

import io
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import pandas as pd
import requests
from tqdm.auto import tqdm

from data.repo_paths import data_cache_dir

logger = logging.getLogger(__name__)

FINRA_DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/{facility}shvol{yyyymmdd}.txt"
DEFAULT_FACILITIES = ("FNSQ", "FNYX", "FNRA")  # exclude FNQC and CNMS
DEFAULT_MAX_WORKERS = 8
_USER_AGENT = (
    "Mozilla/5.0 (compatible; trading_portfolio/0.1; "
    "+https://github.com/local/trading_portfolio)"
)

DEFAULT_CACHE_DIR = data_cache_dir()

_REQUEST_SLEEP_SEC = 0.15
_SOFT_FAIL_RETRIES = 2
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

DayStatus = Literal["ok", "hard_empty", "soft_fail"]

_thread_local = threading.local()


@dataclass
class _DayFetchResult:
    frame: pd.DataFrame
    status: DayStatus


@dataclass
class _FacilityYearState:
    facility: str
    year: int
    cache_dir: str
    frame: pd.DataFrame | None = None
    empty_dates: set[date] = field(default_factory=set)
    covered_months: set[str] = field(default_factory=set)
    frame_loaded: bool = False
    empty_loaded: bool = False
    covered_loaded: bool = False
    frame_dirty: bool = False
    empty_dirty: bool = False
    covered_dirty: bool = False

    @property
    def year_path(self) -> str:
        return _year_cache_path(self.cache_dir, self.facility, self.year)

    @property
    def empty_path(self) -> str:
        return _empty_dates_path(self.cache_dir, self.facility, self.year)

    @property
    def covered_path(self) -> str:
        return _covered_months_path(self.cache_dir, self.facility, self.year)


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


def _month_key(day: pd.Timestamp) -> str:
    return f"{int(day.year):04d}-{int(day.month):02d}"


def _year_cache_path(cache_dir: str, facility: str, year: int) -> str:
    return os.path.join(cache_dir, _CACHE_SUBDIR, f"{facility}_{year}.parquet")


def _empty_dates_path(cache_dir: str, facility: str, year: int) -> str:
    return os.path.join(
        cache_dir, _CACHE_SUBDIR, f"{facility}_{year}_empty.parquet"
    )


def _covered_months_path(cache_dir: str, facility: str, year: int) -> str:
    return os.path.join(
        cache_dir, _CACHE_SUBDIR, f"{facility}_{year}_covered.parquet"
    )


def parse_finra_short_volume_text(text: str) -> pd.DataFrame:
    """Parse a FINRA daily Reg SHO short-volume file body.

    Drops the header, trailer / record-count lines, blank symbols, and rows
    whose numeric fields fail coercion.
    """
    if not text or not str(text).strip():
        return _empty_frame()

    lines: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("date|"):
            continue
        # Trailer is typically a bare record count (no pipes).
        if line.count("|") < 4:
            continue
        lines.append(line)
    if not lines:
        return _empty_frame()

    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep="|",
        header=None,
        dtype=str,
        engine="c",
    )
    if df.shape[1] < 5:
        return _empty_frame()

    raw_ticker = df.iloc[:, 1]
    tickers = raw_ticker.astype(str).str.strip().str.upper()
    ticker_ok = (
        raw_ticker.notna()
        & (tickers != "")
        & (~tickers.isin({"NAN", "NONE", "<NA>"}))
    )
    df = df.loc[ticker_ok].copy()
    if df.empty:
        return _empty_frame()

    if df.shape[1] >= 6:
        short_vol = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        short_exempt = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        total_vol = pd.to_numeric(df.iloc[:, 4], errors="coerce")
    else:
        short_vol = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        short_exempt = pd.Series(0.0, index=df.index)
        total_vol = pd.to_numeric(df.iloc[:, 3], errors="coerce")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(
                df.iloc[:, 0].astype(str).str.strip(),
                format="%Y%m%d",
                errors="coerce",
            ),
            "ticker": df.iloc[:, 1].astype(str).str.strip().str.upper(),
            "short_volume": short_vol,
            "short_exempt_volume": short_exempt,
            "total_volume": total_vol,
        }
    )
    out = out.dropna(subset=["date", "ticker", "short_volume", "total_volume"])
    if out.empty:
        return _empty_frame()
    out["date"] = out["date"].dt.normalize()
    out["short_exempt_volume"] = out["short_exempt_volume"].fillna(0.0)
    return out[list(_RAW_COLS)].reset_index(drop=True)


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


def _thread_session() -> requests.Session:
    """Keep-alive Session local to the current thread (pool-safe)."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = _make_session()
        _thread_local.session = sess
    return sess


def _download_facility_day_once(
    facility: str,
    day: pd.Timestamp,
    *,
    session: requests.Session,
    sleep_sec: float = 0.0,
) -> _DayFetchResult:
    """Single CDN GET. Distinguishes hard-empty (404) from soft-fail."""
    yyyymmdd = day.strftime("%Y%m%d")
    url = FINRA_DAILY_URL.format(facility=facility, yyyymmdd=yyyymmdd)
    try:
        resp = session.get(url, timeout=60)
    except requests.RequestException as exc:
        logger.warning("FINRA request failed %s %s: %s", facility, yyyymmdd, exc)
        return _DayFetchResult(_empty_frame(), "soft_fail")

    if resp.status_code == 404:
        logger.debug("FINRA 404 (empty day): %s %s", facility, yyyymmdd)
        return _DayFetchResult(_empty_frame(), "hard_empty")

    if resp.status_code != 200:
        logger.warning(
            "FINRA HTTP error %s %s: status=%s",
            facility,
            yyyymmdd,
            resp.status_code,
        )
        return _DayFetchResult(_empty_frame(), "soft_fail")

    if sleep_sec > 0:
        time.sleep(sleep_sec)

    body = resp.text
    if not body or not body.strip():
        return _DayFetchResult(_empty_frame(), "hard_empty")

    frame = parse_finra_short_volume_text(body)
    if frame.empty:
        # Successful HTTP but no parseable symbol rows (holiday / empty file).
        return _DayFetchResult(_empty_frame(), "hard_empty")
    return _DayFetchResult(frame, "ok")


def _download_facility_day(
    facility: str,
    day: pd.Timestamp,
    *,
    session: requests.Session | None = None,
    sleep_sec: float = 0.0,
    retries: int = _SOFT_FAIL_RETRIES,
) -> _DayFetchResult:
    """Fetch one facility/day file with soft-fail retries.

    Returns ``_DayFetchResult`` with status ``ok``, ``hard_empty`` (404 /
    empty body), or ``soft_fail`` (network / non-404 HTTP after retries).
    """
    sess = session if session is not None else _thread_session()
    attempts = max(1, int(retries) + 1)
    last = _DayFetchResult(_empty_frame(), "soft_fail")
    for i in range(attempts):
        last = _download_facility_day_once(
            facility, day, session=sess, sleep_sec=sleep_sec
        )
        if last.status != "soft_fail":
            return last
        if i + 1 < attempts:
            time.sleep(0.05 * (i + 1))
    return last


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


def _load_covered_months(path: str) -> set[str]:
    if not os.path.isfile(path):
        return set()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FINRA covered-months cache unreadable %s: %s", path, exc)
        return set()
    if df.empty or "month" not in df.columns:
        return set()
    return {str(m).strip() for m in df["month"].tolist() if str(m).strip()}


def _write_covered_months(path: str, months: set[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not months:
        pd.DataFrame({"month": pd.Series(dtype="object")}).to_parquet(
            path, index=False
        )
        return
    out = pd.DataFrame({"month": sorted(months)})
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


def _ensure_state_sidecars(state: _FacilityYearState) -> None:
    if not state.empty_loaded:
        state.empty_dates = _load_empty_dates(state.empty_path)
        state.empty_loaded = True
    if not state.covered_loaded:
        state.covered_months = _load_covered_months(state.covered_path)
        state.covered_loaded = True


def _ensure_state_frame(state: _FacilityYearState) -> pd.DataFrame:
    if not state.frame_loaded:
        state.frame = _load_year_cache(state.year_path)
        state.frame_loaded = True
    assert state.frame is not None
    return state.frame


def _have_dates(state: _FacilityYearState) -> set[date]:
    _ensure_state_sidecars(state)
    frame = _ensure_state_frame(state)
    have: set[date] = set(state.empty_dates)
    if not frame.empty:
        have |= set(pd.to_datetime(frame["date"]).dt.normalize().dt.date)
    return have


def _get_state(
    memo: dict[tuple[str, int], _FacilityYearState],
    facility: str,
    year: int,
    cache_dir: str,
) -> _FacilityYearState:
    key = (facility, year)
    state = memo.get(key)
    if state is None:
        state = _FacilityYearState(facility=facility, year=year, cache_dir=cache_dir)
        memo[key] = state
    return state


def _download_jobs(
    jobs: list[tuple[str, pd.Timestamp]],
    *,
    max_workers: int,
) -> list[tuple[str, pd.Timestamp, _DayFetchResult]]:
    """Download ``(facility, day)`` jobs; thread-local Sessions when parallel."""
    if not jobs:
        return []

    workers = max(1, int(max_workers))
    sleep_sec = _REQUEST_SLEEP_SEC if workers == 1 else 0.0
    results: list[tuple[str, pd.Timestamp, _DayFetchResult]] = []

    def _one(facility: str, day: pd.Timestamp) -> tuple[str, pd.Timestamp, _DayFetchResult]:
        # Parallel: each worker uses thread-local Session.
        # Sequential polite mode: reuse one Session on this thread.
        sess = _thread_session()
        return facility, day, _download_facility_day(
            facility, day, session=sess, sleep_sec=sleep_sec
        )

    if workers == 1:
        for facility, day in jobs:
            results.append(_one(facility, day))
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, fac, day) for fac, day in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def _apply_download_results(
    memo: dict[tuple[str, int], _FacilityYearState],
    cache_dir: str,
    results: list[tuple[str, pd.Timestamp, _DayFetchResult]],
) -> None:
    # Group new frames / hard empties by facility-year.
    new_frames: dict[tuple[str, int], list[pd.DataFrame]] = {}
    for facility, day, result in results:
        year = int(day.year)
        state = _get_state(memo, facility, year, cache_dir)
        _ensure_state_sidecars(state)
        _ensure_state_frame(state)

        if result.status == "ok":
            new_frames.setdefault((facility, year), []).append(result.frame)
        elif result.status == "hard_empty":
            d = day.date()
            if d not in state.empty_dates:
                state.empty_dates.add(d)
                state.empty_dirty = True
        # soft_fail: leave missing for a later call

    for (facility, year), pieces in new_frames.items():
        state = _get_state(memo, facility, year, cache_dir)
        merged = _merge_year_pieces(_ensure_state_frame(state), pieces)
        state.frame = merged
        state.frame_dirty = True


def _calendar_month_bdays(month_days: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """All business days in the calendar month of ``month_days`` (same month)."""
    if not month_days:
        return []
    anchor = month_days[0]
    m_start = pd.Timestamp(year=int(anchor.year), month=int(anchor.month), day=1)
    m_end = m_start + pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)
    return list(pd.bdate_range(m_start, m_end))


def _maybe_mark_month_covered(
    state: _FacilityYearState,
    month_days: list[pd.Timestamp],
) -> None:
    """Mark YYYY-MM covered when every calendar-month business day is known."""
    if not month_days:
        return
    key = _month_key(month_days[0])
    _ensure_state_sidecars(state)
    if key in state.covered_months:
        return
    have = _have_dates(state)
    for d in _calendar_month_bdays(month_days):
        if d.date() not in have:
            return
    state.covered_months.add(key)
    state.covered_dirty = True


def _flush_state(state: _FacilityYearState) -> None:
    if state.frame_dirty and state.frame is not None:
        _write_year_cache(state.year_path, state.frame)
        state.frame_dirty = False
    if state.empty_dirty:
        _write_empty_dates(state.empty_path, state.empty_dates)
        state.empty_dirty = False
    if state.covered_dirty:
        _write_covered_months(state.covered_path, state.covered_months)
        state.covered_dirty = False


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
        Completed months: ``{FACILITY}_{YYYY}_covered.parquet``.
    max_workers
        Parallel CDN downloads across facilities within each month (default 8).
        Use ``1`` for sequential polite mode (includes short sleep after
        successes).
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

    memo: dict[tuple[str, int], _FacilityYearState] = {}
    years_needed: set[tuple[str, int]] = set()

    for m_start, m_end in tqdm(
        month_windows,
        desc="FINRA short vol",
        unit="month",
        file=sys.stderr,
    ):
        month_days = [d for d in needed if m_start <= d <= m_end]
        if not month_days:
            continue
        year = int(m_start.year)
        key = _month_key(m_start)
        jobs: list[tuple[str, pd.Timestamp]] = []

        for fac in fac_list:
            years_needed.add((fac, year))
            state = _get_state(memo, fac, year, cache_dir)
            _ensure_state_sidecars(state)
            if key in state.covered_months:
                continue
            have = _have_dates(state)
            missing = [d for d in month_days if d.date() not in have]
            for d in missing:
                jobs.append((fac, d))

        if jobs:
            logger.info(
                "FINRA %s: downloading %d missing facility-day(s)",
                key,
                len(jobs),
            )
            results = _download_jobs(jobs, max_workers=max_workers)
            _apply_download_results(memo, cache_dir, results)

        for fac in fac_list:
            state = _get_state(memo, fac, year, cache_dir)
            _maybe_mark_month_covered(state, month_days)

    # Persist dirty caches once per facility-year.
    for state in memo.values():
        _flush_state(state)

    # Assemble return panel: load any covered-only years not yet in memory.
    facility_frames: list[pd.DataFrame] = []
    for fac, year in sorted(years_needed):
        state = _get_state(memo, fac, year, cache_dir)
        year_df = _ensure_state_frame(state)
        if year_df is None or year_df.empty:
            continue
        mask = (year_df["date"] >= start_ts) & (year_df["date"] <= end_ts)
        sliced = year_df.loc[mask]
        if sliced.empty:
            continue
        # Filter to caller tickers before cross-facility concat/groupby.
        filtered = _filter_to_tickers(sliced, tickers)
        if not filtered.empty:
            facility_frames.append(filtered)

    if not facility_frames:
        return _empty_frame()

    combined = pd.concat(facility_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined["ticker"] = combined["ticker"].astype(str).str.upper()

    out = (
        combined.groupby(["date", "ticker"], as_index=False)[
            ["short_volume", "short_exempt_volume", "total_volume"]
        ]
        .sum()
    )
    if out.empty:
        return _empty_frame()
    out = out.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    return out[list(_OUTPUT_COLS)]
