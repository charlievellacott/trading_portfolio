"""S2 live helpers: cache paths, pair-id parsing, long OHLCV → store frames."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from data.ingestion.equity_fetcher import ohlcv_dates_to_naive_utc

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CACHE_DIR = os.path.join(_HERE, "cache")
_PANEL_CACHE_NAME = "s2_live_panel.parquet"
_RETURNS_CACHE_NAME = "s2_live_book_returns.parquet"
_SIZING_STATE_NAME = "s2_live_sizing_state.json"


def cache_dir() -> str:
    override = os.environ.get("S2_CACHE_DIR", "").strip()
    if override:
        return os.path.abspath(override)
    return _DEFAULT_CACHE_DIR


def panel_cache_path() -> str:
    return os.path.join(cache_dir(), _PANEL_CACHE_NAME)


def returns_cache_path() -> str:
    return os.path.join(cache_dir(), _RETURNS_CACHE_NAME)


def sizing_state_path() -> str:
    return os.path.join(cache_dir(), _SIZING_STATE_NAME)


def parse_pair_id(pair_id: str) -> tuple[str, str]:
    parts = str(pair_id).split("|")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"pair_id must be ticker_y|ticker_x, got {pair_id!r}")
    return parts[0], parts[1]


def tickers_from_pair_ids(pair_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pid in pair_ids:
        y, x = parse_pair_id(pid)
        for t in (y, x):
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def long_ohlcv_to_frames(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long ``fetch_ohlcv`` panel → DatetimeIndex OHLC frames for ``build_pair_panel``."""
    out: dict[str, pd.DataFrame] = {}
    if long_df is None or long_df.empty:
        return out
    d = long_df.copy()
    d["date"] = ohlcv_dates_to_naive_utc(d["date"])
    d["ticker"] = d["ticker"].astype(str).str.strip().str.upper()
    for ticker, g in d.groupby("ticker", sort=False):
        frame = g.set_index("date")[["open", "high", "low", "close"]].sort_index()
        out[str(ticker)] = frame
    return out


def save_panel_cache(panel: pd.DataFrame) -> str:
    path = panel_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    panel.to_parquet(path, index=False)
    return path


def save_book_returns_cache(returns: pd.Series) -> str:
    path = returns_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s = pd.Series(returns, dtype=float).sort_index()
    s.index = pd.to_datetime(s.index)
    frame = s.rename("ret").rename_axis("date").reset_index()
    frame.to_parquet(path, index=False)
    return path


def load_sizing_state() -> dict[str, Any]:
    path = sizing_state_path()
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def save_sizing_state(state: dict[str, Any]) -> str:
    path = sizing_state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
        f.write("\n")
    return path
