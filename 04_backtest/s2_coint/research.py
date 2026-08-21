"""Notebook helpers for the S2 hypothesis stack (research runner, not Strategy)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import pandas as pd

from backtest.s2_coint.report import load_star_stack, require_star
from data.ingestion.equity_fetcher import ohlcv_dates_to_naive_utc
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.metrics import load_s1_period_returns

_HERE = os.path.abspath(__file__)
_PKG_DIR = os.path.dirname(_HERE)

BARS_PER_SESSION_1H = 6
DAY_OLS_WINDOW = 252
DAY_Z_WINDOW = 60
DAY_HL_WINDOW = 252
DAY_SIGMA_WINDOW = 60
KALMAN_BURN_IN_DAYS = 30

FROZEN_C_PAIRS: tuple[str, ...] = (
    "1398.HK|0939.HK",
    "1288.HK|3328.HK",
    "8306.T|8316.T",
)
RESEARCH_IS_END_C = "2021-12-31"


def repo_root(start: str | None = None) -> str:
    cur = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isdir(os.path.join(cur, "01_data", "ingestion")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start or os.getcwd())
        cur = parent


# Strict editable installs resolve __file__ under build/; STAR JSON lives in the repo.
ARTIFACTS_DIR = os.path.join(repo_root(_PKG_DIR), "04_backtest", "s2_coint", "artifacts")
DEFAULT_STAR_STACK = os.path.join(ARTIFACTS_DIR, "s2_star_stack.json")


def lookbacks_for_bar(bar: str) -> dict[str, int]:
    """Map day-unit lookbacks to bar counts. 1H uses sessions, not raw hours."""
    if bar == "1h":
        m = BARS_PER_SESSION_1H
        return {
            "ols_window": DAY_OLS_WINDOW * m,
            "z_window": DAY_Z_WINDOW * m,
            "hl_window": DAY_HL_WINDOW * m,
            "sigma_window": DAY_SIGMA_WINDOW * m,
            "kalman_burn_in": KALMAN_BURN_IN_DAYS * m,
        }
    if bar != "1d":
        raise ValueError(f"bar must be '1d' or '1h' (no 4h), got {bar!r}")
    return {
        "ols_window": DAY_OLS_WINDOW,
        "z_window": DAY_Z_WINDOW,
        "hl_window": DAY_HL_WINDOW,
        "sigma_window": DAY_SIGMA_WINDOW,
        "kalman_burn_in": KALMAN_BURN_IN_DAYS,
    }


def half_life_to_sessions(hl, *, bar: str):
    """Convert OU half-life from bar units into sessions for 1D vs 1H compare.

    1d: one bar is one session. 1h: divide by ``BARS_PER_SESSION_1H`` (6).
    """
    if bar == "1h":
        scale = float(BARS_PER_SESSION_1H)
    elif bar == "1d":
        scale = 1.0
    else:
        raise ValueError(f"bar must be '1d' or '1h' (no 4h), got {bar!r}")
    if isinstance(hl, pd.Series):
        return hl.astype(float) / scale
    return float(hl) / scale


def parse_pair_id(pair_id: str) -> tuple[str, str]:
    parts = str(pair_id).split("|")
    if len(parts) != 2:
        raise ValueError(f"pair_id must be ticker_y|ticker_x, got {pair_id!r}")
    return parts[0], parts[1]


def unique_tickers(pair_ids: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for pid in pair_ids:
        y, x = parse_pair_id(pid)
        for t in (y, x):
            if t not in seen:
                seen.append(t)
    return seen


def pair_tuples(pair_ids: Sequence[str]) -> list[tuple[str, str]]:
    return [parse_pair_id(p) for p in pair_ids]


def long_ohlcv_to_frames(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long fetch_ohlcv panel → DatetimeIndex OHLC frames for ``build_pair_panel``.

    Dates are forced to naive UTC via ``ohlcv_dates_to_naive_utc`` (same helper
    as ``_download_ohlcv`` for ``interval='1h'``).
    """
    out: dict[str, pd.DataFrame] = {}
    if long_df.empty:
        return out
    d = long_df.copy()
    # Fetch already stores 1h as naive UTC; re-apply so mixed HK/Tokyo tz
    # never reaches build_pair_panel if a caller skipped fetch_ohlcv.
    d["date"] = ohlcv_dates_to_naive_utc(d["date"])
    for ticker, g in d.groupby("ticker", sort=False):
        frame = g.set_index("date")[["open", "high", "low", "close"]].sort_index()
        out[str(ticker)] = frame
    return out


def filter_pairs(panel: pd.DataFrame, pair_ids: Sequence[str]) -> pd.DataFrame:
    keep = set(pair_ids)
    return panel.loc[panel["pair_id"].isin(keep)].copy()


def s2_data_dir(root: str | None = None) -> str:
    return os.path.join(repo_root(root), "01_data", "data_files", "s2_coint")


def panel_paths(bar: str, *, universe: str = "C", root: str | None = None) -> tuple[str, str]:
    data = s2_data_dir(root)
    train = os.path.join(data, f"s2_panel_{universe}_{bar}_train.parquet")
    full = os.path.join(data, f"s2_panel_{universe}_{bar}_full.parquet")
    return train, full


def load_universe_c_panels(
    bar: str,
    pair_ids: Sequence[str],
    *,
    root: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train (research IS) and full census panels; filter to locked pairs."""
    train_path, full_path = panel_paths(bar, root=root)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"missing {train_path}. For 1D run s2_pair_panel.ipynb; "
            "for 1H run H-002 (no 4H panel exists for universe C)."
        )
    train = filter_pairs(pd.read_parquet(train_path), pair_ids)
    full = (
        filter_pairs(pd.read_parquet(full_path), pair_ids)
        if os.path.isfile(full_path)
        else train.copy()
    )
    train["date"] = pd.to_datetime(train["date"])
    full["date"] = pd.to_datetime(full["date"])
    return train, full


def overlap_calendar_bounds(a: pd.Series, b: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive calendar min/max shared by two timestamp series."""
    a_d = pd.to_datetime(a)
    b_d = pd.to_datetime(b)
    start = max(a_d.min(), b_d.min())
    end = min(a_d.max(), b_d.max())
    if end < start:
        raise ValueError("no overlapping calendar window")
    return pd.Timestamp(start), pd.Timestamp(end)


def clip_panel_calendar(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    d = pd.to_datetime(panel["date"])
    return panel.loc[(d >= start) & (d <= end)].copy()


def split_is_oos(
    panel: pd.DataFrame,
    *,
    is_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.to_datetime(panel["date"])
    return panel.loc[d <= is_end].copy(), panel.loc[d > is_end].copy()


def is_end_for_stack(stack: dict, panel: pd.DataFrame) -> pd.Timestamp:
    """1D uses frozen RESEARCH_IS_END; 1H uses 70% of the available panel dates."""
    bar = stack.get("BAR_STAR")
    require_star("BAR_STAR", bar)
    if bar == "1d":
        return pd.Timestamp(stack.get("RESEARCH_IS_END_STAR") or RESEARCH_IS_END_C)
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"])).sort_values().unique()
    if len(dates) < 10:
        raise ValueError("1H panel too short for a 70/30 IS:OOS split")
    cut = int(len(dates) * 0.70) - 1
    cut = min(max(cut, 0), len(dates) - 2)
    return pd.Timestamp(dates[cut])


def overlap_is_end(panel: pd.DataFrame, *, frac: float = 0.70) -> pd.Timestamp:
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"])).sort_values().unique()
    cut = int(len(dates) * frac) - 1
    cut = min(max(cut, 0), len(dates) - 2)
    return pd.Timestamp(dates[cut])


def _hl_gate(raw) -> tuple[float | None, float | None]:
    if raw in (None, "off", "None"):
        return None, None
    if raw in ("5_60", "[5, 60]", "[5,60]", "5,60"):
        return 5.0, 60.0
    if raw in ("5_30", "[5, 30]", "[5,30]", "5,30"):
        return 5.0, 30.0
    raise ValueError(f"unrecognized HL_GATE_STAR {raw!r}")


def _corr_k(raw) -> float | None:
    if raw in (None, "off", "None"):
        return None
    return float(raw)


def config_from_stack(stack: dict, **overrides) -> S2SimConfig:
    """Build a sim config from frozen STARs; unset knobs keep H-001 defaults."""
    bar = overrides.pop("bar", None) or stack.get("BAR_STAR") or "1d"
    hl_min, hl_max = _hl_gate(stack.get("HL_GATE_STAR"))
    lb = lookbacks_for_bar(str(bar))
    kwargs: dict = {
        "hedge": stack.get("HEDGE_STAR") or "ols",
        "bar": bar,
        "entry_z": float(stack.get("ENTRY_Z_STAR") or 2.0),
        "exit_z": float(stack.get("EXIT_Z_STAR") or 0.0),
        "ols_window": lb["ols_window"],
        "z_window": lb["z_window"],
        "hl_window": lb["hl_window"],
        "sigma_window": lb["sigma_window"],
        "break_mode": stack.get("BREAK_STAR") or "off",
        "trend_mode": stack.get("TREND_STAR") or "off",
        "hl_gate_min": hl_min,
        "hl_gate_max": hl_max,
        "overlap_mode": stack.get("OVERLAP_STAR") or "allow",
        "exit_mode": stack.get("EXIT_STAR") or "mean_only",
        "corr_k": _corr_k(stack.get("CORR_GATE_STAR")),
        "size_mode": stack.get("SIZE_STAR") or "equal",
        "vol_mode": stack.get("VOL_STAR") or "fixed_k",
        "z_window_mode": stack.get("Z_WINDOW_MODE_STAR") or "fixed",
        "entry_mode": stack.get("ENTRY_STAR") or "trad_z",
    }
    kwargs.update(overrides)
    return S2SimConfig(**kwargs)


def load_s1_weekly(root: str | None = None) -> pd.Series:
    path = os.path.join(
        repo_root(root),
        "01_data",
        "data_files",
        "s1_equities",
        "s1_period_returns.parquet",
    )
    return load_s1_period_returns(path)


def overlay_kalman_hedge(
    panel: pd.DataFrame,
    *,
    burn_in: int = 30,
    z_window: int = 60,
    hl_window: int = 252,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
) -> pd.DataFrame:
    """Replace OLS α/β/spread/z/HL with Kalman prior-state hedge (H-003)."""
    from data.processing.s2_coint_store import (
        compute_coint_metrics,
        compute_half_life,
        compute_kalman_hedge_spread,
        compute_spread_zscore,
    )

    if panel.empty:
        return panel.copy()
    parts: list[pd.DataFrame] = []
    for _, g in panel.groupby("pair_id", sort=False):
        g = g.sort_values("date").copy()
        idx = pd.to_datetime(g["date"])
        y = pd.Series(g["close_y"].to_numpy(dtype=float), index=idx)
        x = pd.Series(g["close_x"].to_numpy(dtype=float), index=idx)
        hedge = compute_kalman_hedge_spread(
            y, x, delta=delta, obs_var=obs_var, burn_in=burn_in
        )
        g["alpha"] = hedge["alpha"].to_numpy(dtype=float)
        g["beta"] = hedge["beta"].to_numpy(dtype=float)
        g["spread"] = hedge["spread"].to_numpy(dtype=float)
        g["spread_var"] = hedge["spread_var"].to_numpy(dtype=float)
        g["z_innov"] = hedge["z_innov"].to_numpy(dtype=float)
        g["z"] = compute_spread_zscore(hedge["spread"], window=z_window).to_numpy(dtype=float)
        g["half_life"] = compute_half_life(hedge["spread"], window=hl_window).to_numpy(
            dtype=float
        )
        metrics = compute_coint_metrics(hedge["spread"])
        for col in metrics.columns:
            g[col] = metrics[col].to_numpy(dtype=float)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def overlay_ols_hedge(
    panel: pd.DataFrame,
    *,
    ols_window: int,
    z_window: int = 60,
    hl_window: int = 252,
) -> pd.DataFrame:
    """Replace panel α/β/spread/z/HL with rolling OLS hedge (H-002 session window)."""
    from data.processing.s2_coint_store import (
        compute_coint_metrics,
        compute_half_life,
        compute_spread_zscore,
        compute_static_hedge_spread,
    )

    if panel.empty:
        return panel.copy()
    parts: list[pd.DataFrame] = []
    for _, g in panel.groupby("pair_id", sort=False):
        g = g.sort_values("date").copy()
        idx = pd.to_datetime(g["date"])
        y = pd.Series(g["close_y"].to_numpy(dtype=float), index=idx)
        x = pd.Series(g["close_x"].to_numpy(dtype=float), index=idx)
        hedge = compute_static_hedge_spread(y, x, window=ols_window)
        g["alpha"] = hedge["alpha"].to_numpy(dtype=float)
        g["beta"] = hedge["beta"].to_numpy(dtype=float)
        g["spread"] = hedge["spread"].to_numpy(dtype=float)
        g["z"] = compute_spread_zscore(hedge["spread"], window=z_window).to_numpy(dtype=float)
        g["half_life"] = compute_half_life(hedge["spread"], window=hl_window).to_numpy(
            dtype=float
        )
        metrics = compute_coint_metrics(hedge["spread"])
        for col in metrics.columns:
            g[col] = metrics[col].to_numpy(dtype=float)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def tearsheet_path(hyp_id: str, arm: str) -> str:
    return os.path.join(ARTIFACTS_DIR, f"{hyp_id}_{arm}.pdf")
