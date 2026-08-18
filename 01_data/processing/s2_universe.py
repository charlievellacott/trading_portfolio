"""S2 universe helpers: CSV load and same-venue pair candidacy (not cointegration math)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from itertools import combinations


def load_s2_universes(path: str) -> list[list[str]]:
    """Load exactly three universe lines (A / B / C) from ``s2_universes.csv``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Universe CSV not found: {path}")

    universes: list[list[str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            tickers = [t.strip() for t in raw.split(",") if t.strip()]
            if tickers:
                universes.append(tickers)

    if len(universes) != 3:
        raise ValueError(
            f"Expected 3 universe lines in {path}, found {len(universes)}"
        )
    return universes


def ticker_venue_key(ticker: str) -> str:
    """Same-venue key for candidate pairs (blocks e.g. HK↔JP; not within-venue themes).

    Rules (checked in order):
    - exchange suffix after last ``.`` when present (``.HK`` → ``HK``, ``.T`` → ``T``)
    - Yahoo FX suffix ``=X`` → ``FX``
    - Yahoo crypto suffix ``-USD`` → ``CRYPTO``
    - else full ticker (unknown symbols never silently cross-match)
    """
    t = ticker.strip()
    if not t:
        raise ValueError("ticker must be a non-empty string")

    if "." in t:
        # e.g. 0700.HK, 8306.T — last dotted segment is the exchange
        return t.rsplit(".", 1)[-1].upper()
    if t.endswith("=X"):
        return "FX"
    if t.upper().endswith("-USD"):
        return "CRYPTO"
    return t


def iter_same_venue_pairs(tickers: Sequence[str]) -> list[tuple[str, str]]:
    """Unordered same-venue pairs; legs sorted for stable pre-EG ordering."""
    out: list[tuple[str, str]] = []
    for a, b in combinations(tickers, 2):
        if ticker_venue_key(a) != ticker_venue_key(b):
            continue
        left, right = sorted((a, b))
        out.append((left, right))
    return out
