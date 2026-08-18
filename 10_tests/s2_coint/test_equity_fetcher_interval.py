"""1h interval: cache key includes interval; timestamps are not normalized."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ingestion.equity_fetcher import (
    VALID_OHLCV_INTERVALS,
    _cache_key,
    fetch_ohlcv,
)


def test_cache_key_includes_interval():
    tickers = ["1398.HK"]
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-06-01")
    a = _cache_key(tickers, start, end, "single_1398.HK", interval="1d")
    b = _cache_key(tickers, start, end, "single_1398.HK", interval="1h")
    assert a != b


def test_interval_rejects_4h():
    assert "4h" not in VALID_OHLCV_INTERVALS
    with pytest.raises(ValueError, match="interval"):
        fetch_ohlcv("1398.HK", "2024-01-01", interval="4h", isAsian=True, cache_dir=None)
