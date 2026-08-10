"""Offline tests for GDELT sentiment fetcher helpers (no live BQ/HTTP)."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ingestion.alternative_data.sentiment.gdelt_fetcher import (
    DEFAULT_CACHE_DIR,
    DEFAULT_COMPANY_NAME_MAP,
    GDELT_V2_START,
    _aggregate_daily,
    _combine_daily_panels,
    _detect_ambiguous_aliases,
    _fetch_history_bq,
    _log_map_issue,
    _report_mapping_issues,
    _score_gkg_frame,
    _score_gkg_frame_ref,
    build_aliases,
    clamp_gdelt_v2_start,
    filter_alias_map_for_query,
    is_query_alias,
    iter_month_windows,
    iter_year_windows,
    load_company_name_map,
    match_orgs_to_ticker,
    parse_v2tone,
    scored_chunk_path,
)
from data.ingestion.equity_fetcher import DEFAULT_CACHE_DIR as EQUITY_CACHE_DIR


def test_parse_v2tone_first_field() -> None:
    assert parse_v2tone("1.25,2,3") == pytest.approx(1.25)
    assert parse_v2tone(None) is None
    assert parse_v2tone("bad") is None


def test_default_cache_dir_matches_equity() -> None:
    assert DEFAULT_CACHE_DIR == EQUITY_CACHE_DIR
    assert DEFAULT_CACHE_DIR.replace("\\", "/").endswith("01_data/cache")


def test_build_aliases_and_csv_map() -> None:
    aliases = build_aliases("Apple Inc.", extra=["Apple"])
    assert "Apple Inc." in aliases or "Apple Inc" in aliases
    assert any(a.casefold() == "apple" for a in aliases)
    loaded = load_company_name_map(DEFAULT_COMPANY_NAME_MAP)
    assert "AAPL" in loaded
    assert "Apple" in loaded["AAPL"]


def test_match_orgs_and_aggregate() -> None:
    alias_map = {"AAPL": ["Apple"], "MSFT": ["Microsoft"]}
    hits = match_orgs_to_ticker("Foo;Apple Inc;Bar", alias_map)
    assert hits == ["AAPL"]
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03", "2023-01-03", "2023-01-04"]),
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "tone": [1.0, 3.0, -2.0],
        }
    )
    daily = _aggregate_daily(scored)
    aapl = daily[(daily["ticker"] == "AAPL") & (daily["date"] == "2023-01-03")]
    assert aapl["median_tone"].iloc[0] == pytest.approx(2.0)
    assert aapl["n_articles"].iloc[0] == 2


def test_combine_daily_panels_weighted() -> None:
    a = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-03")],
            "ticker": ["AAPL"],
            "median_tone": [1.0],
            "n_articles": [1],
        }
    )
    b = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-03")],
            "ticker": ["AAPL"],
            "median_tone": [3.0],
            "n_articles": [3],
        }
    )
    out = _combine_daily_panels([a, b])
    row = out[(out["ticker"] == "AAPL") & (out["date"] == "2020-01-03")].iloc[0]
    assert row["n_articles"] == 4
    assert row["median_tone"] == pytest.approx(2.5)


def test_score_gkg_frame_matches_ref() -> None:
    raw = pd.DataFrame(
        {
            "gkg_date": ["20200101T120000", "20200102", "bad", "20200103T000000"],
            "V2Organizations": [
                "Foo;Apple Inc;Bar",
                "Microsoft Corp",
                "Apple",
                "Nobody Here",
            ],
            "V2Tone": ["1.5,0,0", "-2.0,1,1", "9.0,0,0", "0.1,0,0"],
        }
    )
    alias_map = {"AAPL": ["Apple"], "MSFT": ["Microsoft"]}
    kwargs = dict(
        date_col="gkg_date",
        orgs_col="V2Organizations",
        tone_col="V2Tone",
    )
    fast = _score_gkg_frame(raw, alias_map, **kwargs)
    ref = _score_gkg_frame_ref(raw, alias_map, **kwargs)
    fast_s = fast.sort_values(["date", "ticker", "tone"]).reset_index(drop=True)
    ref_s = ref.sort_values(["date", "ticker", "tone"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(fast_s, ref_s, check_dtype=False)


def test_clamp_and_windows() -> None:
    assert clamp_gdelt_v2_start("2010-01-04") == GDELT_V2_START
    assert clamp_gdelt_v2_start("2018-06-01") == "2018-06-01"
    windows = iter_year_windows("2015-02-18", "2016-03-01")
    assert windows == [
        ("2015-02-18", "2015-12-31"),
        ("2016-01-01", "2016-03-01"),
    ]
    assert iter_year_windows("2020-06-01", "2020-01-01") == []
    months = iter_month_windows("2015-02-18", "2015-04-02")
    assert months == [
        ("2015-02-18", "2015-02-28"),
        ("2015-03-01", "2015-03-31"),
        ("2015-04-01", "2015-04-02"),
    ]
    assert iter_month_windows("2020-06-01", "2020-01-01") == []


def test_query_alias_hygiene() -> None:
    assert is_query_alias("Apple")
    assert not is_query_alias("IBM")  # len < 4
    assert not is_query_alias("bank")
    assert not is_query_alias("Group")
    filtered = filter_alias_map_for_query(
        {"IBM": ["IBM", "International Business Machines"], "X": ["bank", "Acme"]}
    )
    assert "IBM" in filtered
    assert "IBM" not in filtered["IBM"]
    assert "International Business Machines" in filtered["IBM"]
    assert "X" in filtered
    assert filtered["X"] == ["Acme"]


def test_scored_chunk_resume(tmp_path) -> None:
    cache_dir = str(tmp_path)
    aliases = ["Apple", "Microsoft"]
    path = scored_chunk_path(
        cache_dir,
        window_start="2020-01-01",
        window_end="2020-01-31",
        aliases=aliases,
    )
    cached = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-03")],
            "ticker": ["AAPL"],
            "median_tone": [1.25],
            "n_articles": [4],
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cached.to_parquet(path, index=False)

    alias_map = {"AAPL": ["Apple"], "MSFT": ["Microsoft"]}
    mock_client = MagicMock()
    with patch(
        "data.ingestion.alternative_data.sentiment.gdelt_fetcher._bq_client",
        return_value=mock_client,
    ) as bq_mock:
        out = _fetch_history_bq(
            alias_map,
            "2020-01-01",
            "2020-01-31",
            cache_dir=cache_dir,
            resume=True,
        )
    bq_mock.assert_not_called()
    mock_client.query.assert_not_called()
    assert len(out) == 1
    assert out["ticker"].iloc[0] == "AAPL"
    assert out["median_tone"].iloc[0] == pytest.approx(1.25)
    assert out["n_articles"].iloc[0] == 4

    # resume=False must hit BigQuery even when cache exists
    empty = pd.DataFrame(
        columns=["date", "ticker", "median_tone", "n_articles"]
    )
    job = MagicMock()
    job.result.return_value.to_dataframe.return_value = empty
    mock_client2 = MagicMock()
    mock_client2.query.return_value = job
    with patch(
        "data.ingestion.alternative_data.sentiment.gdelt_fetcher._bq_client",
        return_value=mock_client2,
    ):
        out2 = _fetch_history_bq(
            alias_map,
            "2020-01-01",
            "2020-01-31",
            cache_dir=cache_dir,
            resume=False,
        )
    mock_client2.query.assert_called()
    assert out2.empty


def test_fetch_history_bq_month_loop_calls_query(tmp_path) -> None:
    cache_dir = str(tmp_path)
    alias_map = {"AAPL": ["Apple"], "MSFT": ["Microsoft"]}
    daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-15")],
            "ticker": ["AAPL"],
            "median_tone": [0.5],
            "n_articles": [2],
        }
    )
    job = MagicMock()
    job.result.return_value.to_dataframe.return_value = daily
    mock_client = MagicMock()
    mock_client.query.return_value = job
    with patch(
        "data.ingestion.alternative_data.sentiment.gdelt_fetcher._bq_client",
        return_value=mock_client,
    ):
        out = _fetch_history_bq(
            alias_map,
            "2020-01-01",
            "2020-02-15",
            cache_dir=cache_dir,
            resume=False,
        )
    # Jan + partial Feb → two month windows
    assert mock_client.query.call_count == 2
    sql = mock_client.query.call_args_list[0].args[0]
    assert "APPROX_QUANTILES" in sql
    assert "STRPOS" in sql
    assert "Apple".casefold() in sql.casefold()
    assert not out.empty
    assert set(out.columns) >= {"date", "ticker", "median_tone", "n_articles"}


def test_ambiguous_detection_and_console(capsys) -> None:
    alias_map = {"C": ["Citi"], "X": ["Citi"]}
    issues = _detect_ambiguous_aliases(alias_map)
    assert any(t == "C" for t, _ in issues)

    alias_df = pd.DataFrame(
        [
            {
                "ticker": "ZZZ",
                "cik": "",
                "title": "Zed Zed Zed Corp",
                "n_aliases": 1,
                "aliases": "Zed Zed Zed Corp",
            }
        ]
    )
    daily = pd.DataFrame(columns=["date", "ticker", "median_tone", "n_articles"])
    _report_mapping_issues(["ZZZ"], alias_df, daily)
    captured = capsys.readouterr().out
    assert "GDELT_MAP\tNOT_FOUND\tZed Zed Zed Corp" in captured
    # OK ticker with hits stays silent
    alias_df2 = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "cik": "",
                "title": "Apple Inc",
                "n_aliases": 1,
                "aliases": "Apple",
            }
        ]
    )
    daily2 = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-03")],
            "ticker": ["AAPL"],
            "median_tone": [0.1],
            "n_articles": [5],
        }
    )
    _report_mapping_issues(["AAPL"], alias_df2, daily2)
    captured2 = capsys.readouterr().out
    assert captured2.strip() == ""

    _log_map_issue("AMBIGUOUS", "Citi")
    assert "GDELT_MAP\tAMBIGUOUS\tCiti" in capsys.readouterr().out
