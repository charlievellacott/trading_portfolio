"""Point-in-time S&P 500 constituent snapshots."""

from __future__ import annotations

import logging
import os
from datetime import date
from urllib.request import urlretrieve

import pandas as pd

logger = logging.getLogger(__name__)

SP500_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
DEFAULT_CSV_NAME = "sp500_historical_components.csv"


def _default_csv_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_files",
        "s1_equities",
    )


def ensure_sp500_csv(path: str | None = None) -> str:
    """Download the historical S&P 500 CSV if it is not already on disk."""
    path = path or os.path.join(_default_csv_dir(), DEFAULT_CSV_NAME)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info("Downloading S&P 500 historical constituents to %s", path)
        urlretrieve(SP500_CSV_URL, path)
    return path


def load_sp500_snapshots(csv_path: str | None = None) -> pd.DataFrame:
    """Load sorted S&P 500 snapshot rows (date, tickers)."""
    path = ensure_sp500_csv(csv_path)
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def constituents_on_or_before(
    as_of: date | str,
    csv_path: str | None = None,
) -> tuple[pd.Timestamp, list[str]]:
    """
    Return S&P 500 members from the latest snapshot on or before ``as_of``.

    Uses merge-asof-backward logic: no information after ``as_of`` is used.
    """
    as_of_ts = pd.Timestamp(as_of).normalize()
    df = load_sp500_snapshots(csv_path)

    earliest = df["date"].min()
    if as_of_ts < earliest:
        raise ValueError(
            f"start_date {as_of_ts.date()} is before the earliest S&P 500 snapshot "
            f"({earliest.date()})."
        )

    eligible = df[df["date"] <= as_of_ts]
    if eligible.empty:
        raise ValueError(f"No S&P 500 snapshot on or before {as_of_ts.date()}.")

    row = eligible.iloc[-1]
    universe_date = pd.Timestamp(row["date"]).normalize()
    tickers = [t.strip() for t in str(row["tickers"]).split(",") if t.strip()]
    return universe_date, tickers

