"""Offline unit tests for economic calendar cleaning / parse_value."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data.ingestion.economic_calendar_fetcher as cal_mod
from data.ingestion.economic_calendar_fetcher import clean_calendar_csv, parse_value


def test_parse_value_edge_cases():
    assert parse_value("150K") == pytest.approx(150_000.0)
    assert parse_value("1.5M") == pytest.approx(1_500_000.0)
    assert parse_value("3.2%") == pytest.approx(3.2)
    assert np.isnan(parse_value("-"))
    assert np.isnan(parse_value(""))
    assert np.isnan(parse_value("<0.1"))
    assert np.isnan(parse_value(None))


def _toy_calendar_csv(path: str) -> None:
    rows = [
        '2007/01/03,13:15,USD,M,"ADP Non-Farm Employment Change",,,-40K,120K,230K',
        '2007/01/04,08:30,USD,H,"Non-Farm Employment Change",,,100K,150K,140K',
        '2007/01/05,10:00,XYZ,L,"Totally Unknown Widget Print",,,1.0,2.0,3.0',
        '2007/01/06,09:00,EUR,H,"Main Refinancing Rate",,,3.2%,3.0%,2.8%',
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def test_clean_small_fixture_csv_and_unknown_printed():
    tmp = os.path.join(ROOT, "10_tests", "s3_fx_trend", "_tmp_cal_clean")
    os.makedirs(tmp, exist_ok=True)
    csv_path = os.path.join(tmp, "economic_calendar.csv")
    unknown_path = os.path.join(tmp, "unknown_events.csv")
    _toy_calendar_csv(csv_path)

    prev_unknown = cal_mod.UNKNOWN_EVENTS_CSV
    cal_mod.UNKNOWN_EVENTS_CSV = unknown_path
    try:
        if os.path.isfile(unknown_path):
            os.remove(unknown_path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            df = clean_calendar_csv(csv_path)
        printed = buf.getvalue()
        assert "unknown calendar event" in printed
        assert "Totally Unknown Widget Print" in printed
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "event_id" in df.columns
        assert (df["currency"] == "USD").sum() == 2
        adp = df.loc[df["event_id"] == "us_adp_nfp"].iloc[0]
        assert adp["actual"] == pytest.approx(-40_000.0)
        assert adp["pit_safe"] is False or adp["pit_safe"] == False
        parquet = os.path.join(tmp, "economic_calendar.parquet")
        assert os.path.isfile(parquet)
        assert os.path.isfile(unknown_path)
    finally:
        cal_mod.UNKNOWN_EVENTS_CSV = prev_unknown
        for name in (
            "economic_calendar.csv",
            "economic_calendar.parquet",
            "unknown_events.csv",
        ):
            p = os.path.join(tmp, name)
            if os.path.isfile(p):
                os.remove(p)
        if os.path.isdir(tmp):
            try:
                os.rmdir(tmp)
            except OSError:
                pass
