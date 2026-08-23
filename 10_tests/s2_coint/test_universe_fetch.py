"""S2 universe fetch smoke test across every universe in ``S2_POOLS``.

Reads nested pools from ``data.processing.s2_universe_pools`` (no fixed universe count),
batch-downloads monthly max history for earliest dates, then smokes recent daily bars via
``fetch_ohlcv``. ``isAsian`` is derived per ticker: exchange-suffixed names (``.HK``, ``.T``,
``.MC``, ``.MI``, ``.AS``, ``.DE``, ``.PA``) keep their dot, while US names - including share
classes such as ``BF.B`` - are hyphenated for Yahoo.

Network test: run manually, not under pytest.

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
from data.processing.s2_universe import load_s2_pools, pool_tickers, ticker_venue_key

UNIVERSE_NAMES = {
    "A": "forex",
    "B": "crypto",
    "C": "asia (shelved)",
    "D": "us share-class twins",
    "E": "us reit sub-sectors",
    "F": "eur large caps",
}
RECENT_LOOKBACK_DAYS = 45


def needs_exchange_suffix(ticker: str) -> bool:
    """True when the dot is an exchange suffix rather than a US share-class letter."""
    return "." in ticker and ticker_venue_key(ticker) != "US"


def to_yahoo(ticker: str) -> str:
    """Canonical -> Yahoo symbol: keep exchange suffixes, hyphenate US share classes."""
    return ticker if needs_exchange_suffix(ticker) else ticker.replace(".", "-")


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

    # Yahoo wants BF-B, not BF.B; exchange-suffixed names keep their dot.
    yahoo_by_canonical = {t: to_yahoo(t) for t in tickers}
    canonical_by_yahoo = {y: c for c, y in yahoo_by_canonical.items()}
    yahoo_tickers = list(yahoo_by_canonical.values())

    try:
        raw = yf.download(
            yahoo_tickers,
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
    if len(yahoo_tickers) == 1:
        only = yahoo_tickers[0]
        out[canonical_by_yahoo[only]] = _min_date_from_frame(raw, only)
        return out

    for yahoo, canonical in canonical_by_yahoo.items():
        out[canonical] = _min_date_from_frame(raw, yahoo)
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
    every = load_s2_pools()
    print(f"universes: {', '.join(sorted(every))}")

    total = 0
    ok_count = 0

    for label in sorted(every):
        tickers = pool_tickers(every[label])
        name = UNIVERSE_NAMES.get(label, label)
        print(f"\nUniverse {label} ({name}): {len(tickers)} tickers")

        earliest = batch_earliest_monthly(tickers)

        for ticker in tickers:
            total += 1
            first = earliest.get(ticker)
            recent_ok = recent_fetch_ok(
                ticker, isAsian=needs_exchange_suffix(ticker)
            )
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
