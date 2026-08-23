"""Regulatory short-selling bans: block new entries on the leg that cannot be shorted.

A ban removes the short leg exactly during the stress windows that produce the widest
``|z|`` - the trades that look best in a naive backtest. Universe F's Iberian, Italian and
French pools are directly exposed, and every window below falls inside F's research IS
(``RESEARCH_IS_END = 2021-12-31``), so this changes H-001 scoring rather than being a
live-only footnote.

Semantics reuse the H-004 demotion rule, so this adds a mask rather than a code path:

- A ban blocks **new entries in the affected direction only**. Long spread needs to short
  ``x``; short spread needs to short ``y``.
- Any open position runs to its normal z-exit. Real bans restricted opening or increasing net
  short positions, not holding, and nothing is force-closed.
- Selection is untouched: a pair can stay cointegrated through a ban, so p-value ranking and
  the caps never see ban logic.

Universes A, B, C, D and E have **no records**, so their masks are all-True and behaviour is
unchanged.

Not modelled: market-maker exemptions and the 2020 net-short-position thresholds (we are not
a market maker), and the SEC's September 2008 ban on ~799 US financials (2008-09-19 ->
2008-10-08). That list was amended repeatedly and its coverage of equity REITs and
share-class twins is uncertain, so asserting membership would be worse than omitting it;
adding a record here is a one-line change.

Dates below must be verified against the AMF / CNMV / CONSOB announcements before H-001
scores F - a few days of error at a window edge changes which trades are blocked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "SHORT_BANS",
    "ShortBan",
    "ban_windows_for_ticker",
    "pair_entry_masks",
    "short_banned",
    "short_banned_series",
]


@dataclass(frozen=True)
class ShortBan:
    """One regulator ban window.

    ``exchange_suffix`` applies the ban to every ticker on that venue (a blanket ban).
    ``tickers`` restricts it to named symbols (a financials-only ban). Exactly one of the
    two is used; ``tickers`` wins when both are given.
    """

    regulator: str
    start: str
    end: str
    source: str
    exchange_suffix: str | None = None
    tickers: tuple[str, ...] = ()

    def covers(self, ticker: str) -> bool:
        t = ticker.strip().upper()
        if self.tickers:
            return t in {x.upper() for x in self.tickers}
        if self.exchange_suffix:
            return t.endswith(self.exchange_suffix.upper())
        return False


# Universe F financial names, used by the 2011-12 financials-only bans.
_ES_FINANCIALS: tuple[str, ...] = (
    "SAN.MC",
    "BBVA.MC",
    "CABK.MC",
    "SAB.MC",
    "BKT.MC",
)
_IT_FINANCIALS: tuple[str, ...] = (
    "ISP.MI",
    "UCG.MI",
    "BAMI.MI",
    "BPE.MI",
    "BMPS.MI",
)

SHORT_BANS: tuple[ShortBan, ...] = (
    # --- 2011-12 sovereign debt crisis: financial stocks only ---
    ShortBan(
        regulator="AMF (France)",
        start="2011-08-12",
        end="2011-11-11",
        source="AMF ban on net short positions in 11 financial stocks; extended to Nov 2011",
        tickers=(),  # no French financials in universe F's fr_luxury pool
    ),
    ShortBan(
        regulator="CONSOB (Italy)",
        start="2011-08-12",
        end="2012-02-24",
        source="CONSOB ban on net short positions in Italian financial stocks",
        tickers=_IT_FINANCIALS,
    ),
    ShortBan(
        regulator="CNMV (Spain)",
        start="2011-08-12",
        end="2012-02-15",
        source="CNMV ban on short selling of Spanish financial stocks",
        tickers=_ES_FINANCIALS,
    ),
    # --- 2012 renewed stress ---
    ShortBan(
        regulator="CONSOB (Italy)",
        start="2012-07-23",
        end="2012-07-27",
        source="CONSOB one-week ban on Italian financial stocks",
        tickers=_IT_FINANCIALS,
    ),
    ShortBan(
        regulator="CNMV (Spain)",
        start="2012-07-23",
        end="2013-01-31",
        source="CNMV blanket ban on all Spanish shares",
        exchange_suffix=".MC",
    ),
    # --- March 2020 COVID: blanket bans. AFM (.AS) and BaFin (.DE) did NOT ban. ---
    ShortBan(
        regulator="CNMV (Spain)",
        start="2020-03-17",
        end="2020-05-18",
        source="CNMV blanket ban on all Spanish shares",
        exchange_suffix=".MC",
    ),
    ShortBan(
        regulator="CONSOB (Italy)",
        start="2020-03-18",
        end="2020-05-18",
        source="CONSOB blanket ban on all Italian shares",
        exchange_suffix=".MI",
    ),
    ShortBan(
        regulator="AMF (France)",
        start="2020-03-18",
        end="2020-05-18",
        source="AMF blanket ban on all French shares",
        exchange_suffix=".PA",
    ),
)


def ban_windows_for_ticker(ticker: str) -> list[ShortBan]:
    """Every ban record covering ``ticker`` (empty for universes A-E)."""
    return [ban for ban in SHORT_BANS if ban.covers(ticker)]


def short_banned(ticker: str, date) -> bool:
    """True when shorting ``ticker`` is prohibited on ``date`` (inclusive window)."""
    d = pd.Timestamp(date)
    for ban in ban_windows_for_ticker(ticker):
        if pd.Timestamp(ban.start) <= d <= pd.Timestamp(ban.end):
            return True
    return False


def short_banned_series(ticker: str, dates: Sequence) -> np.ndarray:
    """Vectorised ``short_banned`` over a date sequence."""
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates))))
    out = np.zeros(len(idx), dtype=bool)
    for ban in ban_windows_for_ticker(ticker):
        out |= (idx >= pd.Timestamp(ban.start)) & (idx <= pd.Timestamp(ban.end))
    return out


def pair_entry_masks(
    ticker_y: str,
    ticker_x: str,
    dates: Sequence,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(long_spread_allowed, short_spread_allowed)`` per bar.

    Long spread is long ``y`` / short ``x``, so it is blocked while ``x`` is banned.
    Short spread is short ``y`` / long ``x``, so it is blocked while ``y`` is banned.
    Both banned means no new entries in either direction.
    """
    y_banned = short_banned_series(ticker_y, dates)
    x_banned = short_banned_series(ticker_x, dates)
    return ~x_banned, ~y_banned
