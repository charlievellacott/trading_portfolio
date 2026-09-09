"""S3 FX price panel: OHLCV long panel + TSMOM / Donchian / realised-vol features."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import date

import pandas as pd

from data.ingestion.fx_fetcher import G10_V1_PAIRS, fetch_fx_ohlcv_batch
from data.repo_paths import repo_root

logger = logging.getLogger(__name__)

RESEARCH_IS_END_S3 = pd.Timestamp("2019-12-31")

_HERE = os.path.abspath(__file__)
_PKG_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_DATA_DIR = os.path.join(_PKG_ROOT, "data_files", "s3_fx_trend")

_OHLC_COLS = ("date", "pair", "open", "high", "low", "close")


def _normalize_pair(pair: str) -> str:
    return str(pair).strip().upper().replace("_", "").replace("/", "").replace("=X", "")


def price_panel_path(interval: str = "1d", *, data_dir: str | None = None) -> str:
    root = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    return os.path.join(root, f"s3_price_panel_{interval}.parquet")


def build_s3_price_panel(
    pairs: Sequence[str] = G10_V1_PAIRS,
    start: date | str | pd.Timestamp = "2007-01-01",
    end: date | str | pd.Timestamp | None = None,
    *,
    interval: str = "1d",
    source: str = "oanda",
    cache: bool = True,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Long FX OHLCV panel: ``date, pair, open, high, low, close``.

    Day boundary follows the FX fetcher (NY 17:00 / naive UTC). When
    ``cache=True``, reads/writes ``s3_price_panel_{interval}.parquet``.
    """
    out_dir = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    cache_path = price_panel_path(interval, data_dir=out_dir)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None

    if cache and os.path.isfile(cache_path):
        cached = pd.read_parquet(cache_path)
        if not cached.empty and set(_OHLC_COLS).issubset(cached.columns):
            cached = cached.copy()
            cached["date"] = pd.to_datetime(cached["date"])
            cached["pair"] = cached["pair"].map(_normalize_pair)
            mask = cached["date"] >= start_ts
            if end_ts is not None:
                mask &= cached["date"] <= end_ts
            hit = cached.loc[mask].copy()
            want = {_normalize_pair(p) for p in pairs}
            hit = hit.loc[hit["pair"].isin(want)]
            if not hit.empty:
                return (
                    hit.loc[:, list(_OHLC_COLS)]
                    .sort_values(["date", "pair"])
                    .reset_index(drop=True)
                )

    raw = fetch_fx_ohlcv_batch(
        pairs,
        start_ts,
        end_ts,
        interval=interval,
        source=source,
    )
    if raw is None or raw.empty:
        empty = pd.DataFrame(columns=list(_OHLC_COLS))
        return empty

    panel = raw.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["pair"] = panel["pair"].map(_normalize_pair)
    for col in ("open", "high", "low", "close"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = (
        panel.loc[:, list(_OHLC_COLS)]
        .dropna(subset=["date", "pair", "close"])
        .sort_values(["date", "pair"])
        .drop_duplicates(subset=["date", "pair"], keep="last")
        .reset_index(drop=True)
    )

    if cache:
        os.makedirs(out_dir, exist_ok=True)
        # Persist full build (not just the requested slice) under the interval key.
        panel.to_parquet(cache_path, index=False)
        logger.info("wrote S3 price panel %s (%d rows)", cache_path, len(panel))

    return panel


def add_tsmom(panel: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Add ``tsmom_{L} = close / close.shift(L) - 1`` per pair (no ``shift(-1)``)."""
    L = int(lookback_days)
    if L < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
    if panel is None or panel.empty:
        out = panel.copy() if panel is not None else pd.DataFrame()
        out[f"tsmom_{L}"] = pd.Series(dtype=float)
        return out

    d = panel.copy()
    d["date"] = pd.to_datetime(d["date"])
    col = f"tsmom_{L}"
    parts: list[pd.DataFrame] = []
    for _, g in d.groupby("pair", sort=False):
        g = g.sort_values("date").copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        g[col] = close / close.shift(L) - 1.0
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_donchian(panel: pd.DataFrame, N: int) -> pd.DataFrame:
    """Donchian channel + break flags using prior-bar channel (no same-bar peek).

    ``donch_high/low_{N}`` = rolling max/min of high/low over ``N`` bars, then
    ``.shift(1)`` so close ``t`` compares to ``max(high_{t-N:t-1})``.
    """
    n = int(N)
    if n < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    if panel is None or panel.empty:
        out = panel.copy() if panel is not None else pd.DataFrame()
        for c in (
            f"donch_high_{n}",
            f"donch_low_{n}",
            f"donch_break_up_{n}",
            f"donch_break_dn_{n}",
        ):
            out[c] = pd.Series(dtype=float)
        return out

    d = panel.copy()
    d["date"] = pd.to_datetime(d["date"])
    hi_col = f"donch_high_{n}"
    lo_col = f"donch_low_{n}"
    up_col = f"donch_break_up_{n}"
    dn_col = f"donch_break_dn_{n}"
    parts: list[pd.DataFrame] = []
    for _, g in d.groupby("pair", sort=False):
        g = g.sort_values("date").copy()
        high = pd.to_numeric(g["high"], errors="coerce")
        low = pd.to_numeric(g["low"], errors="coerce")
        close = pd.to_numeric(g["close"], errors="coerce")
        # Prior channel: rolling(N).max/min then shift(1).
        g[hi_col] = high.rolling(n, min_periods=n).max().shift(1)
        g[lo_col] = low.rolling(n, min_periods=n).min().shift(1)
        g[up_col] = (close > g[hi_col]).astype(float)
        g[dn_col] = (close < g[lo_col]).astype(float)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_realised_vol(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add ``sigma_{W}`` = population std of simple close-to-close returns."""
    W = int(window)
    if W < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if panel is None or panel.empty:
        out = panel.copy() if panel is not None else pd.DataFrame()
        out[f"sigma_{W}"] = pd.Series(dtype=float)
        return out

    d = panel.copy()
    d["date"] = pd.to_datetime(d["date"])
    col = f"sigma_{W}"
    parts: list[pd.DataFrame] = []
    for _, g in d.groupby("pair", sort=False):
        g = g.sort_values("date").copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        rets = close.pct_change(fill_method=None)
        g[col] = rets.rolling(W, min_periods=W).std(ddof=0)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def s3_data_dir(root: str | None = None) -> str:
    """``01_data/data_files/s3_fx_trend`` under the repo root."""
    base = root if root is not None else repo_root()
    return os.path.join(base, "01_data", "data_files", "s3_fx_trend")
