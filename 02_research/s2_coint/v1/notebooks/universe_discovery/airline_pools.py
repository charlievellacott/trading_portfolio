"""Default US-listed airline / carrier pools for discovery notebooks.

Tickers are NYSE/Nasdaq airline (and closely related passenger-aviation) names
that are candidates for Alpaca. Nested leaves keep pairs within theme. The
notebook drops names that fail OHLCV fetch or Alpaca ``tradable`` / ``shortable``.
"""

from __future__ import annotations

# 50 US-listed airline / carrier candidates across sub-sectors.
DEFAULT_AIRLINE_POOLS: dict[str, list[str]] = {
    "us_majors": [
        "DAL",
        "UAL",
        "AAL",
        "LUV",
        "ALK",
        "JBLU",
        "HA",
    ],
    "us_ulcc": [
        "ULCC",
        "ALGT",
        "SNCY",
        "SAVE",
        "FLYY",
    ],
    "us_regional": [
        "SKYW",
        "RJET",
        "MESA",
    ],
    "us_cargo_charter": [
        "ATSG",
        "FLYX",
        "UP",
        "AAWW",
    ],
    "latam": [
        "CPA",
        "VLRS",
        "GOL",
        "AZUL",
        "LTM",
        "AERO",
    ],
    "europe_adr": [
        "RYAAY",
        "ICAGY",
        "DLAKY",
        "AFLYY",
        "EZJZY",
        "NRYAY",
        "FINMY",
    ],
    "asia_adr": [
        "SINGY",
        "CPCAY",
        "JAPSY",
        "ALNPY",
        "AIRYY",
        "CEA",
        "ZNH",
    ],
    "air_taxi_evtol": [
        "JOBY",
        "BLDE",
        "SRFM",
        "SOAR",
        "JTAI",
        "ACHR",
        "EVTL",
        "LILM",
        "EH",
        "PONY",
        "ZK",
    ],
}


def airline_ticker_count(pools: dict | None = None) -> int:
    p = pools if pools is not None else DEFAULT_AIRLINE_POOLS
    seen: set[str] = set()
    for tickers in p.values():
        for t in tickers:
            seen.add(str(t).upper())
    return len(seen)


def filter_pools_to_tickers(
    pools: dict[str, list[str]],
    keep: set[str] | list[str],
) -> dict[str, list[str]]:
    """Drop tickers not in ``keep``; drop empty leaves."""
    allow = {str(t).upper() for t in keep}
    out: dict[str, list[str]] = {}
    for leaf, tickers in pools.items():
        kept = [t for t in tickers if str(t).upper() in allow]
        if len(kept) >= 2:
            out[str(leaf)] = kept
        elif kept:
            # Singleton leaf cannot form a pair — drop.
            continue
    return out
