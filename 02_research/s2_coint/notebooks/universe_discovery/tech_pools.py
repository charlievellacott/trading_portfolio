"""US-listed tech universe builder for discovery notebooks.

Loads a Nasdaq screener snapshot (cached), keeps Technology + Telecommunications
names, ranks by market cap (cap ``TARGET_N``), and shards industries into leaf
pools so within-leaf pair counts stay tractable for the method.md screen.
"""

from __future__ import annotations

import os
import re
from urllib.request import urlretrieve

import pandas as pd

# Prefer listed tech + telecom (US exchanges). Full Technology+Telecom is ~700;
# TARGET_N caps at 1000 when a broader snapshot is available.
TARGET_N = 1000
MAX_LEAF = 25
TECH_SECTORS = frozenset({"Technology", "Telecommunications"})
# Industry substrings used to catch tech names filed under adjacent sectors.
TECH_INDUSTRY_SUBSTR = (
    "software",
    "semiconductor",
    "computer",
    "edp services",
    "electronic component",
    "telecommunications equipment",
    "communications equipment",
    "computer peripheral",
    "computer manufacturing",
)
TICKERS_CSV_URL = (
    "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/"
    "data/v2/tickers.csv"
)

_HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CACHE_DIR = os.path.join(_HERE, "cache")
DEFAULT_CACHE_CSV = os.path.join(DEFAULT_CACHE_DIR, "us_nasdaq_screener_tickers.csv")

__all__ = [
    "DEFAULT_CACHE_CSV",
    "MAX_LEAF",
    "TARGET_N",
    "TECH_SECTORS",
    "ensure_screener_csv",
    "filter_pools_to_tickers",
    "load_us_tech_pools",
    "tech_ticker_count",
]


def ensure_screener_csv(
    path: str | None = None,
    *,
    force: bool = False,
) -> str:
    """Download Nasdaq screener ticker CSV if missing (or ``force``)."""
    path = path or DEFAULT_CACHE_CSV
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if force or not os.path.isfile(path):
        urlretrieve(TICKERS_CSV_URL, path)
    return path


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_").lower()
    return s or "unknown"


def filter_pools_to_tickers(
    pools: dict[str, list[str]],
    keep: set[str] | list[str],
) -> dict[str, list[str]]:
    """Drop tickers not in ``keep``; drop leaves with fewer than 2 names."""
    allow = {str(t).upper() for t in keep}
    out: dict[str, list[str]] = {}
    for leaf, tickers in pools.items():
        kept = [t for t in tickers if str(t).upper() in allow]
        if len(kept) >= 2:
            out[str(leaf)] = kept
    return out


def tech_ticker_count(pools: dict | None = None) -> int:
    if pools is None:
        pools = load_us_tech_pools()
    seen: set[str] = set()
    for tickers in pools.values():
        for t in tickers:
            seen.add(str(t).upper())
    return len(seen)


def load_us_tech_frame(
    csv_path: str | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Raw screener rows for tech + telecom, sorted by market cap descending."""
    path = ensure_screener_csv(csv_path, force=refresh)
    df = pd.read_csv(path)
    need = {"symbol", "sector", "industry", "market_cap"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"screener CSV missing columns: {sorted(missing)}")

    d = df.copy()
    d["symbol"] = d["symbol"].astype(str).str.upper().str.strip()
    d["sector"] = d["sector"].astype(str)
    d["industry"] = d["industry"].astype(str)
    d["market_cap"] = pd.to_numeric(d["market_cap"], errors="coerce")

    # Prefer common stock; drop blanks / warrants-ish / preferred share suffixes.
    d = d.loc[d["symbol"].str.len().between(1, 5)]
    d = d.loc[~d["symbol"].str.contains(r"[^A-Z]", na=False)]
    if "name" in d.columns:
        name_l = d["name"].astype(str).str.lower()
        drop_name = name_l.str.contains(
            r"warrant|right|unit|preferred|depositary|etf|fund",
            na=False,
        )
        d = d.loc[~drop_name]
    ind_l = d["industry"].astype(str).str.lower()
    ind_hit = False
    for sub in TECH_INDUSTRY_SUBSTR:
        ind_hit = ind_hit | ind_l.str.contains(re.escape(sub), na=False)
    sector_hit = d["sector"].isin(TECH_SECTORS)
    d = d.loc[sector_hit | ind_hit]
    d = d.dropna(subset=["symbol"])
    # Prefer names with a usable market cap for ranking; keep NaN cap at the end.
    d = d.sort_values(
        ["market_cap"],
        ascending=False,
        kind="mergesort",
        na_position="last",
    )
    d = d.drop_duplicates(subset=["symbol"], keep="first")
    return d.reset_index(drop=True)


def load_us_tech_pools(
    *,
    target_n: int = TARGET_N,
    max_leaf: int = MAX_LEAF,
    csv_path: str | None = None,
    refresh: bool = False,
) -> dict[str, list[str]]:
    """Nested industry (sharded) pools of US-listed tech / telecom names.

    Returns at most ``target_n`` tickers (by market cap). Large industries are
    split into shards of size ``max_leaf`` so ``iter_pool_pairs`` stays usable.
    """
    if target_n < 2:
        raise ValueError("target_n must be >= 2")
    if max_leaf < 2:
        raise ValueError("max_leaf must be >= 2")

    d = load_us_tech_frame(csv_path, refresh=refresh)
    if d.empty:
        raise RuntimeError("no Technology/Telecommunications rows in screener CSV")

    d = d.head(int(target_n)).copy()
    pools: dict[str, list[str]] = {}
    for industry, g in d.groupby("industry", sort=False):
        tickers = g["symbol"].astype(str).tolist()
        base = _slug(industry)
        if len(tickers) <= int(max_leaf):
            if len(tickers) >= 2:
                pools[base] = tickers
            continue
        shard = 0
        for i in range(0, len(tickers), int(max_leaf)):
            chunk = tickers[i : i + int(max_leaf)]
            if len(chunk) < 2:
                # Merge singleton remainder into previous shard when possible.
                if chunk and pools:
                    last_key = f"{base}_{shard - 1}" if shard else base
                    if last_key in pools:
                        pools[last_key].extend(chunk)
                    continue
                break
            key = f"{base}_{shard}"
            pools[key] = chunk
            shard += 1
    return pools
