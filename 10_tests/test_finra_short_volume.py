"""Tests for FINRA daily short-volume fetcher (H-010; no live HTTP)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd

from data.ingestion.alternative_data.finra_short_volume import (
    _canonical_finra_ticker,
    _empty_dates_path,
    _ticker_aliases,
    _year_cache_path,
    fetch_short_volume_daily,
    parse_finra_short_volume_text,
)


def _fixture_pipe_text() -> str:
    """Header, normal rows, malformed row, blank symbol, footer."""
    return "\n".join(
        [
            "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
            "20240102|AAPL|100|5|400|Q",
            "20240102|BRK/B|50|0|200|Q",
            "20240102||10|0|20|Q",  # blank symbol
            "20240102|BAD|x|0|10|Q",  # malformed numeric
            "not_a_row",
            "20240102|MSFT|20|1|80|Q",
            "4",  # trailer / record count
        ]
    )


def test_parse_drops_footer_and_bad_rows() -> None:
    df = parse_finra_short_volume_text(_fixture_pipe_text())
    assert list(df.columns) == [
        "date",
        "ticker",
        "short_volume",
        "short_exempt_volume",
        "total_volume",
    ]
    assert set(df["ticker"]) == {"AAPL", "BRK/B", "MSFT"}
    assert len(df) == 3
    aapl = df.loc[df["ticker"] == "AAPL"].iloc[0]
    assert aapl["short_volume"] == 100.0
    assert aapl["short_exempt_volume"] == 5.0
    assert aapl["total_volume"] == 400.0
    assert pd.Timestamp(aapl["date"]) == pd.Timestamp("2024-01-02")


def test_canonical_and_aliases() -> None:
    assert _canonical_finra_ticker(" brk.b ") == "BRK.B"
    aliases = _ticker_aliases("BRK.B")
    assert "BRK.B" in aliases
    assert "BRK/B" in aliases
    assert "BRKB" in aliases


def test_sum_across_facilities_from_cache(tmp_path) -> None:
    cache_dir = str(tmp_path)
    sub = os.path.join(cache_dir, "finra_short_volume")
    os.makedirs(sub, exist_ok=True)

    day = pd.Timestamp("2024-01-02")
    fnsq = pd.DataFrame(
        {
            "date": [day, day],
            "ticker": ["AAPL", "BRK/B"],
            "short_volume": [100.0, 50.0],
            "short_exempt_volume": [5.0, 0.0],
            "total_volume": [400.0, 200.0],
        }
    )
    fnyx = pd.DataFrame(
        {
            "date": [day, day],
            "ticker": ["AAPL", "MSFT"],
            "short_volume": [30.0, 10.0],
            "short_exempt_volume": [1.0, 2.0],
            "total_volume": [90.0, 40.0],
        }
    )
    fnsq.to_parquet(os.path.join(sub, "FNSQ_2024.parquet"), index=False)
    fnyx.to_parquet(os.path.join(sub, "FNYX_2024.parquet"), index=False)

    out = fetch_short_volume_daily(
        ["AAPL", "BRK.B", "MSFT"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        facilities=("FNSQ", "FNYX"),
        cache_dir=cache_dir,
    )

    assert list(out.columns) == [
        "date",
        "ticker",
        "short_volume",
        "short_exempt_volume",
        "total_volume",
    ]
    assert set(out["ticker"]) == {"AAPL", "BRK.B", "MSFT"}

    aapl = out.loc[out["ticker"] == "AAPL"].iloc[0]
    assert aapl["short_volume"] == 130.0
    assert aapl["short_exempt_volume"] == 6.0
    assert aapl["total_volume"] == 490.0

    brk = out.loc[out["ticker"] == "BRK.B"].iloc[0]
    assert brk["short_volume"] == 50.0
    assert brk["total_volume"] == 200.0

    msft = out.loc[out["ticker"] == "MSFT"].iloc[0]
    assert msft["short_volume"] == 10.0


def test_empty_when_nothing_found(tmp_path) -> None:
    cache_dir = str(tmp_path)
    sub = os.path.join(cache_dir, "finra_short_volume")
    os.makedirs(sub, exist_ok=True)
    day = pd.Timestamp("2024-01-02")
    pd.DataFrame(
        {
            "date": [day],
            "ticker": ["ZZZZ"],
            "short_volume": [1.0],
            "short_exempt_volume": [0.0],
            "total_volume": [2.0],
        }
    ).to_parquet(os.path.join(sub, "FNSQ_2024.parquet"), index=False)

    out = fetch_short_volume_daily(
        ["AAPL"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        facilities=("FNSQ",),
        cache_dir=cache_dir,
    )
    assert out.empty
    assert list(out.columns) == [
        "date",
        "ticker",
        "short_volume",
        "short_exempt_volume",
        "total_volume",
    ]


def test_empty_day_sidecar_skips_second_download(tmp_path) -> None:
    cache_dir = str(tmp_path)
    day = pd.Timestamp("2024-01-02")
    calls: list[pd.Timestamp] = []

    def _fake_download(facility, d, *, session=None, sleep_sec=0.0):
        calls.append(pd.Timestamp(d).normalize())
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "short_volume",
                "short_exempt_volume",
                "total_volume",
            ]
        )

    with patch(
        "data.ingestion.alternative_data.finra_short_volume._download_facility_day",
        side_effect=_fake_download,
    ):
        out1 = fetch_short_volume_daily(
            ["AAPL"],
            start_date="2024-01-02",
            end_date="2024-01-02",
            facilities=("FNSQ",),
            cache_dir=cache_dir,
            max_workers=1,
        )
        n_first = len(calls)
        out2 = fetch_short_volume_daily(
            ["AAPL"],
            start_date="2024-01-02",
            end_date="2024-01-02",
            facilities=("FNSQ",),
            cache_dir=cache_dir,
            max_workers=1,
        )

    assert out1.empty and out2.empty
    assert n_first == 1
    assert len(calls) == 1  # second call did not hit network
    empty_path = _empty_dates_path(cache_dir, "FNSQ", 2024)
    assert os.path.isfile(empty_path)
    empty = pd.read_parquet(empty_path)
    assert pd.Timestamp(empty["date"].iloc[0]).normalize() == day


def test_parallel_download_sums_facilities(tmp_path) -> None:
    cache_dir = str(tmp_path)
    day = pd.Timestamp("2024-01-03")

    def _fake_download(facility, d, *, session=None, sleep_sec=0.0):
        vol = 100.0 if facility == "FNSQ" else 40.0
        return pd.DataFrame(
            {
                "date": [pd.Timestamp(d).normalize()],
                "ticker": ["AAPL"],
                "short_volume": [vol],
                "short_exempt_volume": [1.0],
                "total_volume": [vol * 4],
            }
        )

    with patch(
        "data.ingestion.alternative_data.finra_short_volume._download_facility_day",
        side_effect=_fake_download,
    ):
        out = fetch_short_volume_daily(
            ["AAPL"],
            start_date="2024-01-03",
            end_date="2024-01-03",
            facilities=("FNSQ", "FNYX"),
            cache_dir=cache_dir,
            max_workers=4,
        )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["short_volume"] == 140.0
    assert row["total_volume"] == 560.0
    assert os.path.isfile(_year_cache_path(cache_dir, "FNSQ", 2024))
    assert os.path.isfile(_year_cache_path(cache_dir, "FNYX", 2024))


def test_cache_hit_month_loop_no_downloads(tmp_path) -> None:
    cache_dir = str(tmp_path)
    sub = os.path.join(cache_dir, "finra_short_volume")
    os.makedirs(sub, exist_ok=True)
    day = pd.Timestamp("2024-02-01")
    pd.DataFrame(
        {
            "date": [day],
            "ticker": ["AAPL"],
            "short_volume": [10.0],
            "short_exempt_volume": [0.0],
            "total_volume": [40.0],
        }
    ).to_parquet(os.path.join(sub, "FNSQ_2024.parquet"), index=False)

    with patch(
        "data.ingestion.alternative_data.finra_short_volume._download_facility_day",
        side_effect=AssertionError("should not download"),
    ) as mock_dl:
        out = fetch_short_volume_daily(
            ["AAPL"],
            start_date="2024-02-01",
            end_date="2024-02-01",
            facilities=("FNSQ",),
            cache_dir=cache_dir,
        )
        mock_dl.assert_not_called()

    assert len(out) == 1
    assert out.iloc[0]["short_volume"] == 10.0
