"""
Economic calendar CSV loader for S3 FX research.

Reads the research calendar at
``01_data/data_files/s3_fx_trend/economic_calendar.csv`` (NO HEADER).

Columns (Forex Factory–style export)::

    Date, Time, Currency, Impact, Event, empty, empty, Actual, Forecast, Previous

Example::

    2007/01/03,13:15,USD,M,"ADP Non-Farm Employment Change",,,-40K,120K,230K

**WARNING — NOT point-in-time / NOT vintage-safe.**
This CSV reflects a later scrapebook of Actual / Forecast / Previous values.
Revised prints and contemporaneous forecast vintages are **not** preserved.
Use for research / diagnostics only; do **not** treat rows as PIT event
features in a live or production backtest without a vintage-safe source.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from typing import Any

import pandas as pd

from data.repo_paths import repo_root

logger = logging.getLogger(__name__)

DEFAULT_CSV = os.path.join(
    repo_root(),
    "01_data",
    "data_files",
    "s3_fx_trend",
    "economic_calendar.csv",
)
DEFAULT_PARQUET = os.path.join(
    os.path.dirname(DEFAULT_CSV),
    "economic_calendar.parquet",
)
UNKNOWN_EVENTS_CSV = os.path.join(
    os.path.dirname(DEFAULT_CSV),
    "unknown_events.csv",
)

_IMPACT_MAP = {"H": 3, "M": 2, "L": 1, "N": 0}

# (currency_upper, normalized event name) → stable event_id
EVENT_NAME_MAP: dict[tuple[str, str], str] = {
    # --- USD ---
    ("USD", "non farm employment change"): "us_nfp",
    ("USD", "adp non farm employment change"): "us_adp_nfp",
    ("USD", "unemployment rate"): "us_unemployment",
    ("USD", "unemployment claims"): "us_jobless_claims",
    ("USD", "average hourly earnings m m"): "us_ahe_mom",
    ("USD", "cpi y y"): "us_cpi_yoy",
    ("USD", "cpi m m"): "us_cpi_mom",
    ("USD", "core cpi m m"): "us_core_cpi_mom",
    ("USD", "core cpi y y"): "us_core_cpi_yoy",
    ("USD", "ppi m m"): "us_ppi_mom",
    ("USD", "core ppi m m"): "us_core_ppi_mom",
    ("USD", "core pce price index m m"): "us_core_pce_mom",
    ("USD", "pce price index m m"): "us_pce_mom",
    ("USD", "retail sales m m"): "us_retail_sales_mom",
    ("USD", "core retail sales m m"): "us_core_retail_sales_mom",
    ("USD", "ism manufacturing pmi"): "us_ism_mfg",
    ("USD", "ism services pmi"): "us_ism_services",
    ("USD", "gdp q q"): "us_gdp_qoq",
    ("USD", "advance gdp q q"): "us_gdp_qoq",
    ("USD", "preliminary gdp q q"): "us_gdp_qoq",
    ("USD", "final gdp q q"): "us_gdp_qoq",
    ("USD", "federal funds rate"): "us_fomc_rate",
    ("USD", "fomc statement"): "us_fomc_statement",
    ("USD", "fomc press conference"): "us_fomc_press",
    ("USD", "fomc meeting minutes"): "us_fomc_minutes",
    ("USD", "trade balance"): "us_trade_balance",
    ("USD", "durable goods orders m m"): "us_durable_goods_mom",
    ("USD", "core durable goods orders m m"): "us_core_durable_goods_mom",
    ("USD", "industrial production m m"): "us_ind_prod_mom",
    ("USD", "cb consumer confidence"): "us_cb_confidence",
    # --- EUR ---
    ("EUR", "main refinancing rate"): "eu_ecb_rate",
    ("EUR", "deposit facility rate"): "eu_ecb_deposit",
    ("EUR", "ecb press conference"): "eu_ecb_press",
    ("EUR", "cpi flash estimate y y"): "eu_cpi_flash_yoy",
    ("EUR", "final cpi y y"): "eu_cpi_yoy",
    ("EUR", "core cpi flash estimate y y"): "eu_core_cpi_flash_yoy",
    ("EUR", "gdp q q"): "eu_gdp_qoq",
    ("EUR", "flash manufacturing pmi"): "eu_mfg_pmi",
    ("EUR", "flash services pmi"): "eu_services_pmi",
    ("EUR", "german flash manufacturing pmi"): "de_mfg_pmi",
    ("EUR", "german prelim cpi m m"): "de_cpi_mom",
    ("EUR", "german ifo business climate"): "de_ifo",
    ("EUR", "retail sales m m"): "eu_retail_sales_mom",
    ("EUR", "unemployment rate"): "eu_unemployment",
    # --- GBP ---
    ("GBP", "official bank rate"): "gb_boe_rate",
    ("GBP", "mpc official bank rate votes"): "gb_mpc_votes",
    ("GBP", "cpi y y"): "gb_cpi_yoy",
    ("GBP", "cpi m m"): "gb_cpi_mom",
    ("GBP", "retail sales m m"): "gb_retail_sales_mom",
    ("GBP", "claimant count change"): "gb_claimant_count",
    ("GBP", "unemployment rate"): "gb_unemployment",
    ("GBP", "average earnings index 3m y"): "gb_avg_earnings",
    ("GBP", "prelim gdp q q"): "gb_gdp_qoq",
    ("GBP", "final manufacturing pmi"): "gb_mfg_pmi",
    ("GBP", "final services pmi"): "gb_services_pmi",
    ("GBP", "goods trade balance"): "gb_trade_balance",
    # --- JPY ---
    ("JPY", "monetary policy statement"): "jp_boj_statement",
    ("JPY", "boj policy rate"): "jp_boj_rate",
    ("JPY", "boj press conference"): "jp_boj_press",
    ("JPY", "tokyo core cpi y y"): "jp_tokyo_core_cpi_yoy",
    ("JPY", "national core cpi y y"): "jp_core_cpi_yoy",
    ("JPY", "gdp q q"): "jp_gdp_qoq",
    ("JPY", "trade balance"): "jp_trade_balance",
    ("JPY", "retail sales y y"): "jp_retail_sales_yoy",
    # --- AUD ---
    ("AUD", "cash rate"): "au_rba_rate",
    ("AUD", "rba rate statement"): "au_rba_statement",
    ("AUD", "cpi q q"): "au_cpi_qoq",
    ("AUD", "trimmed mean cpi q q"): "au_trimmed_cpi_qoq",
    ("AUD", "gdp q q"): "au_gdp_qoq",
    ("AUD", "employment change"): "au_employment",
    ("AUD", "unemployment rate"): "au_unemployment",
    ("AUD", "retail sales m m"): "au_retail_sales_mom",
    ("AUD", "trade balance"): "au_trade_balance",
    # --- CAD ---
    ("CAD", "overnight rate"): "ca_boc_rate",
    ("CAD", "boc rate statement"): "ca_boc_statement",
    ("CAD", "employment change"): "ca_employment",
    ("CAD", "unemployment rate"): "ca_unemployment",
    ("CAD", "cpi m m"): "ca_cpi_mom",
    ("CAD", "core cpi m m"): "ca_core_cpi_mom",
    ("CAD", "gdp m m"): "ca_gdp_mom",
    ("CAD", "retail sales m m"): "ca_retail_sales_mom",
    ("CAD", "trade balance"): "ca_trade_balance",
    # --- NZD ---
    ("NZD", "official cash rate"): "nz_rbnz_rate",
    ("NZD", "rbnz rate statement"): "nz_rbnz_statement",
    ("NZD", "cpi q q"): "nz_cpi_qoq",
    ("NZD", "gdp q q"): "nz_gdp_qoq",
    ("NZD", "employment change"): "nz_employment",
    ("NZD", "unemployment rate"): "nz_unemployment",
    ("NZD", "trade balance"): "nz_trade_balance",
    # --- CHF ---
    ("CHF", "snb policy rate"): "ch_snb_rate",
    ("CHF", "libor rate"): "ch_snb_rate",
    ("CHF", "snb monetary policy assessment"): "ch_snb_assessment",
    ("CHF", "cpi m m"): "ch_cpi_mom",
    ("CHF", "cpi y y"): "ch_cpi_yoy",
}


def parse_value(s: Any) -> float:
    """Parse calendar numeric strings (%, K/M/B/T); ``-`` / empty / ``<`` / ``~`` → NaN."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return float("nan")
    text = str(s).strip()
    if not text or text in {"-", "—", "–"}:
        return float("nan")
    if text.startswith("<") or text.startswith("~") or text.startswith("≈"):
        return float("nan")

    text = text.replace(",", "").replace(" ", "")
    mult = 1.0
    if text.endswith("%"):
        text = text[:-1]
        # keep as percent points (e.g. 0.2% → 0.2), not decimal fraction
    else:
        suffix = text[-1].upper() if text[-1:].isalpha() else ""
        if suffix in {"K", "M", "B", "T"}:
            mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suffix]
            text = text[:-1]

    text = text.lstrip("+")
    try:
        return float(text) * mult
    except ValueError:
        return float("nan")


def _normalize_event_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("y/y", "y y").replace("m/m", "m m").replace("q/q", "q q")
    s = s.replace("3m/y", "3m y")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _slug_event_id(currency: str, event_name: str) -> str:
    ccy = currency.strip().lower()
    slug = _normalize_event_name(event_name).replace(" ", "_")
    if not slug:
        slug = "unknown"
    return f"{ccy}_{slug}"


def _infer_unit(raw: str) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text in {"-", "—", "–"}:
        return None
    if "%" in text:
        return "pct"
    last = text[-1].upper() if text else ""
    if last in {"K", "M", "B", "T"}:
        return last.lower()
    return None


def _load_unknown_event_keys() -> set[tuple[str, str]]:
    existing: set[tuple[str, str]] = set()
    if not os.path.isfile(UNKNOWN_EVENTS_CSV):
        return existing
    with open(UNKNOWN_EVENTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0] != "currency":
                existing.add((row[0], row[1]))
    return existing


def _record_unknown_event(
    currency: str,
    event_name: str,
    *,
    existing: set[tuple[str, str]],
    printed: set[tuple[str, str]],
) -> None:
    """Print unique unknowns this run; append new pairs to unknown_events.csv."""
    key = (currency, event_name)
    if key not in printed:
        msg = f"unknown calendar event: {currency} | {event_name}"
        print(msg, flush=True)
        logger.info(msg)
        printed.add(key)
    if key in existing:
        return
    os.makedirs(os.path.dirname(UNKNOWN_EVENTS_CSV) or ".", exist_ok=True)
    write_header = (
        not os.path.isfile(UNKNOWN_EVENTS_CSV)
        or os.path.getsize(UNKNOWN_EVENTS_CSV) == 0
    )
    with open(UNKNOWN_EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["currency", "event_name"])
        writer.writerow([currency, event_name])
    existing.add(key)


def _localize_to_naive_utc(date_s: str, time_s: str, source_tz: str) -> pd.Timestamp:
    """Combine calendar date+time in ``source_tz`` → naive UTC timestamp."""
    date_s = date_s.strip().replace("-", "/")
    time_s = (time_s or "00:00").strip()
    if not time_s or time_s.lower() in {"all day", "tentative", "tbd"}:
        time_s = "00:00"
    if not re.match(r"^\d{1,2}:\d{2}", time_s):
        time_s = "00:00"
    local = pd.Timestamp(f"{date_s} {time_s}")
    try:
        aware = local.tz_localize(
            source_tz,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    except (TypeError, ValueError):
        try:
            aware = local.tz_localize(
                source_tz,
                ambiguous=True,
                nonexistent="shift_forward",
            )
        except Exception:
            aware = local.tz_localize("UTC")
    utc = aware.tz_convert("UTC")
    return utc.tz_localize(None)


def clean_calendar_csv(
    path: str = DEFAULT_CSV,
    source_tz: str = "US/Eastern",
) -> pd.DataFrame:
    """
    Clean the research calendar CSV into a typed DataFrame and write parquet.

    Returns columns:
    ``timestamp, date, currency, impact, event_name, event_id,
    actual, forecast, previous, unit, pit_safe, source``.

    ``pit_safe`` is always False — see module docstring.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"economic calendar CSV not found: {path}")

    rows: list[dict[str, Any]] = []
    unknown_existing = _load_unknown_event_keys()
    unknown_printed: set[tuple[str, str]] = set()
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw or len(raw) < 5:
                continue
            while len(raw) < 10:
                raw.append("")
            date_s, time_s, currency, impact_s, event_name = (
                raw[0],
                raw[1],
                raw[2],
                raw[3],
                raw[4],
            )
            actual_s, forecast_s, previous_s = raw[7], raw[8], raw[9]
            currency = str(currency).strip().upper()
            event_name = str(event_name).strip()
            if not currency or not event_name:
                continue

            try:
                ts = _localize_to_naive_utc(date_s, time_s, source_tz)
            except Exception as exc:
                logger.warning("skip bad timestamp %s %s: %s", date_s, time_s, exc)
                continue

            impact = _IMPACT_MAP.get(str(impact_s).strip().upper(), 0)
            norm = _normalize_event_name(event_name)
            event_id = EVENT_NAME_MAP.get((currency, norm))
            if event_id is None:
                event_id = _slug_event_id(currency, event_name)
                _record_unknown_event(
                    currency,
                    event_name,
                    existing=unknown_existing,
                    printed=unknown_printed,
                )

            unit = (
                _infer_unit(actual_s)
                or _infer_unit(forecast_s)
                or _infer_unit(previous_s)
            )
            rows.append(
                {
                    "timestamp": ts,
                    "date": ts.normalize(),
                    "currency": currency,
                    "impact": impact,
                    "event_name": event_name,
                    "event_id": event_id,
                    "actual": parse_value(actual_s),
                    "forecast": parse_value(forecast_s),
                    "previous": parse_value(previous_s),
                    "unit": unit,
                    "pit_safe": False,
                    "source": "forexfactory_csv",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("clean_calendar_csv produced empty frame from %s", path)
        return df

    df = df.sort_values(["event_id", "timestamp"], kind="mergesort")
    # Dedup (event_id, timestamp): keep last non-null actual when possible
    df["_actual_rank"] = df["actual"].notna().astype(int)
    df = df.sort_values(
        ["event_id", "timestamp", "_actual_rank"],
        kind="mergesort",
    )
    df = df.drop_duplicates(subset=["event_id", "timestamp"], keep="last")
    df = df.drop(columns=["_actual_rank"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    out_parquet = os.path.join(os.path.dirname(path), "economic_calendar.parquet")
    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    logger.info(
        "wrote %s rows → %s (%d unique event_ids)",
        len(df),
        out_parquet,
        df["event_id"].nunique(),
    )
    return df


def load_economic_calendar(
    path: str | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load cleaned calendar; rebuild from CSV when ``refresh`` or parquet missing."""
    csv_path = path if path is not None else DEFAULT_CSV
    if path is not None and str(path).endswith(".parquet"):
        parquet_path = path
        csv_path = DEFAULT_CSV
    else:
        parquet_path = os.path.join(
            os.path.dirname(csv_path),
            "economic_calendar.parquet",
        )

    if not refresh and os.path.isfile(parquet_path):
        logger.debug("loading economic calendar parquet %s", parquet_path)
        return pd.read_parquet(parquet_path)

    return clean_calendar_csv(csv_path)


def list_event_ids(cal: pd.DataFrame) -> list[str]:
    """Sorted unique ``event_id`` values."""
    if cal is None or cal.empty or "event_id" not in cal.columns:
        return []
    return sorted(cal["event_id"].dropna().astype(str).unique().tolist())


def load_thisweek_calendar(path: str) -> pd.DataFrame:
    """
    Load a live this-week JSON calendar if present.

    The production thisweek.json adapter is deferred. If ``path`` does not
    exist, raises ``NotImplementedError``. If it exists, parses a minimal JSON
    list into the same schema as ``clean_calendar_csv``.
    """
    if not os.path.isfile(path):
        raise NotImplementedError(
            "live thisweek.json adapter is deferred; "
            f"no file at {path}"
        )

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        items = payload.get("events") or payload.get("data") or []
    else:
        items = payload

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        currency = str(
            item.get("currency") or item.get("ccy") or ""
        ).upper()
        event_name = str(
            item.get("event_name")
            or item.get("title")
            or item.get("event")
            or ""
        )
        if not currency or not event_name:
            continue
        ts_raw = item.get("timestamp") or item.get("datetime") or item.get("date")
        if ts_raw is None:
            continue
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        impact_raw = item.get("impact", 0)
        if isinstance(impact_raw, str):
            impact = _IMPACT_MAP.get(impact_raw.strip().upper(), 0)
        else:
            impact = int(impact_raw)
        norm = _normalize_event_name(event_name)
        event_id = EVENT_NAME_MAP.get((currency, norm)) or _slug_event_id(
            currency, event_name
        )
        actual_s = item.get("actual")
        forecast_s = item.get("forecast")
        previous_s = item.get("previous")
        rows.append(
            {
                "timestamp": ts,
                "date": ts.normalize(),
                "currency": currency,
                "impact": impact,
                "event_name": event_name,
                "event_id": event_id,
                "actual": parse_value(actual_s),
                "forecast": parse_value(forecast_s),
                "previous": parse_value(previous_s),
                "unit": item.get("unit"),
                "pit_safe": False,
                "source": os.path.basename(path),
            }
        )
    return pd.DataFrame(rows)
