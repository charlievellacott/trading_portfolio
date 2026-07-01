"""Integration tests for equity data fetchers (live Yahoo Finance).
Uses pytest to test the equity fetcher, to test run: python -m pytest tests/test_fetchers.py -v"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingestion.equity_fetcher import OHLCV_COLUMNS, fetch_top_n_equities

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

TEST_N = 3
TEST_START = "2023-01-03"
TEST_END = "2023-03-31"
MAX_BUSINESS_DAY_GAP = 3


# ---------------------------------------------------------------------------
# Helper subroutines
# ---------------------------------------------------------------------------


def max_business_day_gap(dates: pd.DatetimeIndex) -> int:
    """Largest number of missing business days between consecutive observations."""
    if len(dates) < 2:
        return 0

    ordered = pd.DatetimeIndex(sorted(dates.unique()))
    gaps = [
        max(0, int(np.busday_count(ordered[i - 1].date(), ordered[i].date())) - 1)
        for i in range(1, len(ordered))
    ]
    return max(gaps)


def assert_panel_schema(panel: pd.DataFrame) -> None:
    assert list(panel.columns) == ["date", "ticker", *OHLCV_COLUMNS]


def assert_date_range(
    panel: pd.DataFrame,
    start: str,
    end: str,
) -> None:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    panel_dates = pd.to_datetime(panel["date"]).dt.normalize()

    assert panel_dates.min() >= start_ts, (
        f"earliest row {panel_dates.min().date()} is before start {start_ts.date()}"
    )
    assert panel_dates.max() <= end_ts, (
        f"latest row {panel_dates.max().date()} is after end {end_ts.date()}"
    )


def assert_no_long_date_gaps(
    panel: pd.DataFrame,
    max_gap: int,
) -> None:
    offenders: list[str] = []
    for ticker, group in panel.groupby("ticker"):
        gap = max_business_day_gap(pd.DatetimeIndex(group["date"]))
        if gap > max_gap:
            offenders.append(f"{ticker} (gap={gap} business days)")

    assert not offenders, (
        f"date gaps exceed {max_gap} business days: {', '.join(offenders)}"
    )


def assert_ohlcv_quality(panel: pd.DataFrame) -> None:
    assert not panel.empty, "panel is empty"

    nan_close = panel["close"].isna().sum()
    assert nan_close == 0, f"close has {nan_close} NaN rows"

    for col in ("open", "high", "low", "close"):
        assert (panel[col] > 0).all(), f"{col} must be strictly positive"

    assert (panel["volume"] >= 0).all(), "volume must be non-negative"

    assert panel["ticker"].nunique() == TEST_N, (
        f"expected {TEST_N} tickers, got {panel['ticker'].nunique()}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def equity_panel() -> pd.DataFrame:
    return fetch_top_n_equities(
        TEST_N,
        TEST_START,
        end_date=TEST_END,
        cache_dir=None,
    )

# tests the cols
def test_fetch_top_n_equities_schema(equity_panel: pd.DataFrame) -> None:
    assert_panel_schema(equity_panel)

# tests the date range
def test_fetch_top_n_equities_date_range(equity_panel: pd.DataFrame) -> None:
    assert_date_range(equity_panel, TEST_START, TEST_END)

# tests the max business day gap
def test_fetch_top_n_equities_no_long_gaps(equity_panel: pd.DataFrame) -> None:
    assert_no_long_date_gaps(equity_panel, MAX_BUSINESS_DAY_GAP)

# tests the OHLCV quality: -ve prices, NaN close, negative volume, wrong number of tickers
def test_fetch_top_n_equities_ohlcv_quality(equity_panel: pd.DataFrame) -> None:
    assert_ohlcv_quality(equity_panel)
