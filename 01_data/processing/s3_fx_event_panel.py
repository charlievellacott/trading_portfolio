"""S3 FX event panel: surprise / z-score + quote-convention helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from data.ingestion.fx_fetcher import G10_V1_PAIRS
from data.repo_paths import repo_root

logger = logging.getLogger(__name__)

_HERE = os.path.abspath(__file__)
_PKG_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_DATA_DIR = os.path.join(_PKG_ROOT, "data_files", "s3_fx_trend")
DEFAULT_EVENT_PANEL_PATH = os.path.join(DEFAULT_DATA_DIR, "s3_event_panel.parquet")

_EVENT_COLS = (
    "date",
    "event_id",
    "currency",
    "impact",
    "actual",
    "forecast",
    "surprise",
    "sigma_N",
    "z",
    "unit",
)


def _normalize_pair(pair: str) -> str:
    return str(pair).strip().upper().replace("_", "").replace("/", "").replace("=X", "")


def pair_sign_for_currency(pair: str, ccy: str) -> int:
    """+1 if stronger ``ccy`` profits a long on ``pair``; -1 if quote leg; else 0.

    Examples: EURUSD + EUR → +1; USDJPY + JPY → -1 (JPY up ⇒ USDJPY down).
    """
    p = _normalize_pair(pair)
    c = str(ccy).strip().upper()
    if len(p) != 6 or not c:
        return 0
    base, quote = p[:3], p[3:]
    if c == base:
        return 1
    if c == quote:
        return -1
    return 0


def g10_pair_signs(ccy: str, pairs: Sequence[str] = G10_V1_PAIRS) -> dict[str, int]:
    """Map each v1 pair to ``pair_sign_for_currency`` for ``ccy`` (zeros omitted)."""
    out: dict[str, int] = {}
    for p in pairs:
        s = pair_sign_for_currency(p, ccy)
        if s != 0:
            out[_normalize_pair(p)] = s
    return out


def map_event_to_next_bar_fill(
    event_ts: pd.Timestamp | str,
    bar_index: pd.DatetimeIndex | Sequence[pd.Timestamp] | pd.Series,
    *,
    bar: str = "1d",
) -> pd.Timestamp:
    """First bar timestamp on ``bar_index`` strictly after ``event_ts``.

    ``bar`` is retained for API parity with E-003 (1d / 1h grids); the fill
    rule is always the first open timestamp ``> event_ts``.
    """
    _ = bar  # documented for callers; index already encodes the grid
    tau = pd.Timestamp(event_ts)
    idx = pd.DatetimeIndex(pd.to_datetime(list(bar_index))).sort_values().unique()
    later = idx[idx > tau]
    if len(later) == 0:
        return pd.NaT
    return pd.Timestamp(later[0])


def build_s3_event_panel(
    cal: pd.DataFrame,
    *,
    N: int = 20,
    cache: bool = True,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Build surprise / rolling-sigma / z panel from a cleaned calendar.

    PIT sigma: per ``event_id``, ``surprise.shift(1).rolling(N).std()`` so row
    ``t`` never uses its own surprise in ``sigma_N``.
    """
    n = int(N)
    if n < 2:
        raise ValueError(f"N must be >= 2, got {N}")

    out_dir = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    cache_path = os.path.join(out_dir, "s3_event_panel.parquet")

    if cal is None or cal.empty:
        empty = pd.DataFrame(columns=list(_EVENT_COLS))
        return empty

    d = cal.copy()
    # Prefer print timestamp when present; else calendar ``date``.
    if "timestamp" in d.columns:
        d["date"] = pd.to_datetime(d["timestamp"])
    else:
        d["date"] = pd.to_datetime(d["date"])

    for col in ("actual", "forecast"):
        if col not in d.columns:
            d[col] = np.nan
        else:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d["surprise"] = d["actual"] - d["forecast"]
    if "event_id" not in d.columns:
        raise ValueError("calendar requires event_id")
    if "currency" not in d.columns:
        d["currency"] = ""
    if "impact" not in d.columns:
        d["impact"] = 0
    if "unit" not in d.columns:
        d["unit"] = ""

    d = d.sort_values(["event_id", "date"], kind="mergesort").reset_index(drop=True)

    def _rolling_sigma(s: pd.Series) -> pd.Series:
        # Prior surprises only; rolling window expands until N history exists.
        return s.shift(1).rolling(n, min_periods=2).std(ddof=1)

    d["sigma_N"] = d.groupby("event_id", sort=False)["surprise"].transform(_rolling_sigma)
    sigma = d["sigma_N"].astype(float)
    surprise = d["surprise"].astype(float)
    z = surprise / sigma
    z = z.where(sigma > 0)
    d["z"] = z

    out = d.loc[
        :,
        [
            "date",
            "event_id",
            "currency",
            "impact",
            "actual",
            "forecast",
            "surprise",
            "sigma_N",
            "z",
            "unit",
        ],
    ].copy()
    out["impact"] = pd.to_numeric(out["impact"], errors="coerce").fillna(0).astype(int)
    out["currency"] = out["currency"].astype(str).str.upper()
    out["event_id"] = out["event_id"].astype(str)
    out = out.sort_values("date", kind="mergesort").reset_index(drop=True)

    if cache:
        os.makedirs(out_dir, exist_ok=True)
        out.to_parquet(cache_path, index=False)
        logger.info("wrote S3 event panel %s (%d rows)", cache_path, len(out))

    return out


def s3_event_panel_path(root: str | None = None) -> str:
    base = root if root is not None else repo_root()
    return os.path.join(
        base, "01_data", "data_files", "s3_fx_trend", "s3_event_panel.parquet"
    )
