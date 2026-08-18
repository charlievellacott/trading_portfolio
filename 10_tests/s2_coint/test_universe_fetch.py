"""S2 universe fetch smoke test (FX / crypto / Asia).

Reads ``01_data/data_files/s2_coint/s2_universes.csv`` (3 lines, Yahoo symbols),
batch-downloads monthly max history for earliest dates, then smokes recent
daily bars via ``fetch_ohlcv`` (``isAsian=True`` on line 3).

Run:
    python 10_tests/s2_coint/test_universe_fetch.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ingestion.equity_fetcher import fetch_ohlcv
from data.processing.s2_universe import load_s2_universes
from data.repo_paths import repo_root

UNIVERSE_NAMES = ("forex", "crypto", "asia")
CSV_REL = os.path.join("01_data", "data_files", "s2_coint", "s2_universes.csv")
RECENT_LOOKBACK_DAYS = 45


def universes_csv_path() -> str:
    return os.path.join(repo_root(), CSV_REL)


def _min_date_from_frame(raw: pd.DataFrame, ticker: str) -> pd.Timestamp | None:
    """Earliest valid close date for one ticker from a yfinance download frame."""
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if ticker not in level0:
            return None
        sub = raw[ticker].copy()
    else:
        # Single-ticker download: flat OHLCV columns
        sub = raw.copy()

    if "Close" not in sub.columns:
        return None
    closes = sub["Close"].dropna()
    if closes.empty:
        return None
    idx = closes.index.min()
    return pd.Timestamp(idx).normalize()


def batch_earliest_monthly(tickers: list[str]) -> dict[str, pd.Timestamp | None]:
    """One Yahoo call per universe: monthly max history → earliest date per ticker."""
    out: dict[str, pd.Timestamp | None] = {t: None for t in tickers}
    if not tickers:
        return out

    try:
        raw = yf.download(
            tickers,
            period="max",
            interval="1mo",
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        print(f"  ERROR yf.download failed: {exc}")
        return out

    if raw is None or raw.empty:
        return out

    # Single ticker → flat columns; multi → MultiIndex by ticker
    if len(tickers) == 1:
        out[tickers[0]] = _min_date_from_frame(raw, tickers[0])
        return out

    for ticker in tickers:
        out[ticker] = _min_date_from_frame(raw, ticker)
    return out


def recent_fetch_ok(ticker: str, *, isAsian: bool) -> bool:
    end = datetime.now().date()
    start = end - timedelta(days=RECENT_LOOKBACK_DAYS)
    try:
        panel = fetch_ohlcv(
            ticker,
            start_date=start,
            end_date=end,
            isAsian=isAsian,
        )
    except Exception as exc:
        print(f"  ERROR fetch_ohlcv({ticker}): {exc}")
        return False
    return panel is not None and not panel.empty


def run() -> int:
    path = universes_csv_path()
    print(f"CSV: {path}")
    universes = load_s2_universes(path)

    total = 0
    ok_count = 0

    for i, tickers in enumerate(universes):
        name = UNIVERSE_NAMES[i] if i < len(UNIVERSE_NAMES) else f"line{i + 1}"
        is_asian = name == "asia"
        print(f"\nUniverse {i + 1} ({name}):")

        earliest = batch_earliest_monthly(tickers)

        for ticker in tickers:
            total += 1
            first = earliest.get(ticker)
            recent_ok = recent_fetch_ok(ticker, isAsian=is_asian)
            hist_ok = first is not None
            row_ok = hist_ok and recent_ok
            if row_ok:
                ok_count += 1

            first_s = first.date().isoformat() if first is not None else "NONE"
            print(
                f"  {ticker:12}  earliest={first_s}  "
                f"recent_ok={recent_ok}  ok={row_ok}"
            )

    print(f"\nSUMMARY: {ok_count}/{total} ok")
    return 0 if ok_count == total and total > 0 else 1


if __name__ == "__main__":
    sys.exit(run())
