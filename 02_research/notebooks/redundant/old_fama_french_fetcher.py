"""
ARCHIVED — Ken French Data Library (Dartmouth) ZIP fetcher.

Superseded by ETF Tier A proxies in
``data.ingestion.alternative_data.fama_french_fetcher`` because the Ken French
library is monthly-lagged, historically revised, and therefore not point-in-time
for live / train–serve parity. Kept here for learning and audit — not for
production imports.

Active path: ``01_data/ingestion/alternative_data/fama_french_fetcher.py``
Shim: ``01_data/ingestion/fama_french_fetcher.py``
"""

from __future__ import annotations

import io
import logging
import os
import zipfile

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)
_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Momentum_Factor_daily_CSV.zip"
)

# alternative_data/ -> ingestion/ -> 01_data/ -> cache/
# (path convention from when this lived under alternative_data/)
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "cache",
)

_CACHE_FILE_FF3 = "ff3_daily.parquet"
_CACHE_FILE_MOM = "mom_daily.parquet"


def _download_zip_csv(url: str) -> str:
    """Download a ZIP from the given URL and return the CSV content inside."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        return zf.read(csv_name).decode("utf-8", errors="replace")


def _parse_ff3_csv(raw: str) -> pd.DataFrame:
    """
    Parse the FF3 daily CSV (strip banner/annual section, YYYYMMDD index,
    decimalize percentage values).
    """
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip().replace(" ", "")
        if stripped.startswith("Mkt-RF") or stripped.startswith(",Mkt-RF"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate FF3 CSV header row")

    data_lines = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)

    csv_text = lines[header_idx].strip() + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_text))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date_raw"})
    df["date_raw"] = df["date_raw"].astype(str).str.strip()
    df = df[df["date_raw"].str.len() == 8].copy()
    df["date"] = pd.to_datetime(df["date_raw"], format="%Y%m%d")
    df = df.drop(columns=["date_raw"])

    rename_map = {}
    for col in df.columns:
        low = col.strip().lower().replace("-", "_").replace(" ", "_")
        rename_map[col] = low
    df = df.rename(columns=rename_map)

    for col in ["mkt_rf", "smb", "hml", "rf"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    return df[["date", "mkt_rf", "smb", "hml", "rf"]].sort_values("date").reset_index(drop=True)


def _parse_mom_csv(raw: str) -> pd.DataFrame:
    """Parse Momentum factor daily CSV (same layout conventions)."""
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip().replace(" ", "")
        if "Mom" in line or "WML" in line or stripped.startswith(",Mom"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate Momentum CSV header row")

    data_lines = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)

    csv_text = lines[header_idx].strip() + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_text))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date_raw"})
    df["date_raw"] = df["date_raw"].astype(str).str.strip()
    df = df[df["date_raw"].str.len() == 8].copy()
    df["date"] = pd.to_datetime(df["date_raw"], format="%Y%m%d")
    df = df.drop(columns=["date_raw"])

    mom_col = [c for c in df.columns if c.lower().strip() not in ("date",)][0]
    df["mom"] = pd.to_numeric(df[mom_col], errors="coerce") / 100.0
    return df[["date", "mom"]].sort_values("date").reset_index(drop=True)


def fetch_ff_factors_daily(
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    include_momentum: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Fetch Fama-French daily factors (FF3 + optional Momentum).

    Values are in decimal form (e.g. 0.01 = 1%). Cached to parquet under
    ``cache_dir``.

    Returns DataFrame with columns: date, mkt_rf, smb, hml, [mom,] rf.
    """
    os.makedirs(cache_dir, exist_ok=True)
    ff3_path = os.path.join(cache_dir, _CACHE_FILE_FF3)
    mom_path = os.path.join(cache_dir, _CACHE_FILE_MOM)

    # FF3
    if os.path.exists(ff3_path):
        logger.debug("FF3 cache hit: %s", ff3_path)
        ff3 = pd.read_parquet(ff3_path)
    else:
        logger.info("Downloading FF3 daily factors from Dartmouth...")
        raw = _download_zip_csv(_FF3_URL)
        ff3 = _parse_ff3_csv(raw)
        ff3.to_parquet(ff3_path, index=False)
        logger.info("FF3 cached: %s (%d rows)", ff3_path, len(ff3))

    # Momentum
    if include_momentum:
        if os.path.exists(mom_path):
            logger.debug("Momentum cache hit: %s", mom_path)
            mom = pd.read_parquet(mom_path)
        else:
            logger.info("Downloading Momentum factor from Dartmouth...")
            raw = _download_zip_csv(_MOM_URL)
            mom = _parse_mom_csv(raw)
            mom.to_parquet(mom_path, index=False)
            logger.info("Momentum cached: %s (%d rows)", mom_path, len(mom))

        ff3 = ff3.merge(mom, on="date", how="left")

    # Date filtering
    if start_date is not None:
        ff3 = ff3[ff3["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        ff3 = ff3[ff3["date"] <= pd.Timestamp(end_date)]

    return ff3.sort_values("date").reset_index(drop=True)
