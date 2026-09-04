"""Notebook helpers for the S2 hypothesis stack (research runner, not Strategy)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import pandas as pd

from backtest.star_stack_io import (
    load_star_stack,
    require_star,
    vt_target_ann_vol_from_stack,
)
from data.ingestion.equity_fetcher import ohlcv_dates_to_naive_utc
from data.processing.s2_universe import (
    RESEARCH_IS_END_BY_UNIVERSE,
    SHELVED_UNIVERSES,
)
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

# Universe C locked book. Kept for the Asia postmortem only - C is shelved.
FROZEN_C_PAIRS: tuple[str, ...] = (
    "1398.HK|0939.HK",
    "1288.HK|3328.HK",
    "8306.T|8316.T",
)
RESEARCH_IS_END_C = RESEARCH_IS_END_BY_UNIVERSE["C"]

# Universes shelved / failed at H-001; never selected as UNIVERSE_STAR.
SHELVED = SHELVED_UNIVERSES


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


def lookbacks_for_bar(
    bar: str,
    *,
    ols_days: int | None = None,
    z_days: int | None = None,
    adf_days: int | None = None,
    hl_days: int | None = None,
    sigma_days: int | None = None,
) -> dict[str, int]:
    """Map day-unit lookbacks to bar counts. 1H uses sessions, not raw hours.

    Optional ``*_days`` overrides honor frozen star-stack day units
    (``OLS_WINDOW_STAR``, ``Z_WINDOW_STAR``, ``ADF_WINDOW_STAR``).
    """
    ols = int(DAY_OLS_WINDOW if ols_days is None else ols_days)
    z = int(DAY_Z_WINDOW if z_days is None else z_days)
    adf = int(DAY_OLS_WINDOW if adf_days is None else adf_days)
    hl = int(DAY_HL_WINDOW if hl_days is None else hl_days)
    sigma = int(DAY_SIGMA_WINDOW if sigma_days is None else sigma_days)
    if bar == "1h":
        m = BARS_PER_SESSION_1H
        return {
            "ols_window": ols * m,
            "z_window": z * m,
            "adf_window": adf * m,
            "hl_window": hl * m,
            "sigma_window": sigma * m,
            "kalman_burn_in": KALMAN_BURN_IN_DAYS * m,
        }
    if bar != "1d":
        raise ValueError(f"bar must be '1d' or '1h' (no 4h), got {bar!r}")
    return {
        "ols_window": ols,
        "z_window": z,
        "adf_window": adf,
        "hl_window": hl,
        "sigma_window": sigma,
        "kalman_burn_in": KALMAN_BURN_IN_DAYS,
    }


def _day_star(stack: dict, key: str, default: int) -> int:
    raw = stack.get(key)
    if raw in (None, "None", ""):
        return int(default)
    return int(raw)


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


def candidates_path(bar: str, *, universe: str, root: str | None = None) -> str:
    """Per-candidate screen metrics at IS end (every candidate, not just the selected)."""
    return os.path.join(s2_data_dir(root), f"s2_candidates_{universe}_{bar}.csv")


def pairs_path(bar: str, *, universe: str, root: str | None = None) -> str:
    """Ranked frozen book for a universe (written by H-001)."""
    return os.path.join(s2_data_dir(root), f"s2_pairs_{universe}_{bar}.csv")


def rebalance_path(bar: str, *, universe: str, root: str | None = None) -> str:
    """Quarterly rotating selections (written when BOOK_SCOPE includes rotate)."""
    return os.path.join(s2_data_dir(root), f"s2_rebalance_{universe}_{bar}.csv")


def research_is_end_for(universe: str) -> str:
    """Pre-registered research-IS end for a universe letter (D/E/F = 2021-12-31)."""
    key = str(universe).strip().upper()
    if key not in RESEARCH_IS_END_BY_UNIVERSE:
        raise KeyError(f"no RESEARCH_IS_END registered for universe {universe!r}")
    return RESEARCH_IS_END_BY_UNIVERSE[key]


def frozen_pairs_for_universe(
    universe: str,
    bar: str = "1d",
    *,
    root: str | None = None,
) -> list[str]:
    """Ranked frozen book from ``s2_pairs_{universe}_{bar}.csv`` (best p-value first)."""
    path = pairs_path(bar, universe=universe, root=root)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing {path}. Run H-001_universes.ipynb to write the ranked book."
        )
    df = pd.read_csv(path)
    if "rank" in df.columns:
        df = df.sort_values("rank")
    elif "pvalue" in df.columns:
        df = df.sort_values("pvalue")
    return [str(p) for p in df["pair_id"]]


def load_universe_panels(
    label: str,
    bar: str,
    pair_ids: Sequence[str],
    *,
    root: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train (research IS) and full census panels for any universe; filter to pairs."""
    train_path, full_path = panel_paths(bar, universe=label, root=root)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"missing {train_path}. For 1D run s2_pair_panel.ipynb; "
            "for 1H run H-002 (no 4H panel is built)."
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


def load_universe_c_panels(
    bar: str,
    pair_ids: Sequence[str],
    *,
    root: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Universe C panels (Asia postmortem). Thin alias for ``load_universe_panels("C", ...)``."""
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
    """1D uses frozen RESEARCH_IS_END; 1H uses 70% of the available panel dates.

    On 1D the date comes from ``RESEARCH_IS_END_STAR`` when present, else the registered end
    for ``UNIVERSE_STAR`` (D/E/F = 2021-12-31), else universe C's for the Asia postmortem.
    """
    bar = stack.get("BAR_STAR")
    require_star("BAR_STAR", bar)
    if bar == "1d":
        explicit = stack.get("RESEARCH_IS_END_STAR")
        if explicit:
            return pd.Timestamp(explicit)
        universe = stack.get("UNIVERSE_STAR")
        if universe:
            return pd.Timestamp(research_is_end_for(str(universe)))
        return pd.Timestamp(RESEARCH_IS_END_C)
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
    lb = lookbacks_for_bar(
        str(bar),
        ols_days=_day_star(stack, "OLS_WINDOW_STAR", DAY_OLS_WINDOW),
        z_days=_day_star(stack, "Z_WINDOW_STAR", DAY_Z_WINDOW),
        adf_days=_day_star(stack, "ADF_WINDOW_STAR", DAY_OLS_WINDOW),
    )
    kwargs: dict = {
        "hedge": stack.get("HEDGE_STAR") or "ols",
        "bar": bar,
        "entry_z": float(stack.get("ENTRY_Z_STAR") or 2.0),
        "exit_z": float(stack.get("EXIT_Z_STAR") or 0.0),
        "ols_window": lb["ols_window"],
        "z_window": lb["z_window"],
        "hl_window": lb["hl_window"],
        "adf_window": lb["adf_window"],
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
        "cost_profile": stack.get("COST_PROFILE_STAR")
        or default_cost_profile_for_universe(stack.get("UNIVERSE_STAR")),
        "vt_target_ann_vol": vt_target_ann_vol_from_stack(stack),
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
    adf_window: int = 252,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
) -> pd.DataFrame:
    """Replace OLS α/β/spread/z/HL with Kalman prior-state hedge (H-004)."""
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
        metrics = compute_coint_metrics(hedge["spread"], adf_window=adf_window)
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
    adf_window: int = 252,
) -> pd.DataFrame:
    """Replace panel α/β/spread/z/HL/ADF metrics with rolling OLS hedge."""
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
        metrics = compute_coint_metrics(hedge["spread"], adf_window=adf_window)
        for col in metrics.columns:
            g[col] = metrics[col].to_numpy(dtype=float)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def tearsheet_path(hyp_id: str, arm: str) -> str:
    return os.path.join(ARTIFACTS_DIR, f"{hyp_id}_{arm}.pdf")


# Option 4 validation tiers (see s2_hypothesis_log.md).
HYPOTHESIS_TIERS: dict[str, str] = {
    "H-001": "C",
    "H-004": "C",
    "H-005": "C",
    "H-006": "B",
    "H-002": "A",
    "H-003": "A",
    "H-007": "A",
    "H-008": "A",
    "H-009": "A",
    "H-010": "A",
    "H-011": "A",
    "H-012": "B",
    "H-013": "B",
    "H-015": "B",
}


def hypothesis_tier(hyp_id: str) -> str:
    """Return validation tier A/B/C for a hypothesis id (raises if unknown)."""
    key = str(hyp_id).strip().upper()
    if key not in HYPOTHESIS_TIERS:
        raise KeyError(f"no validation tier registered for {hyp_id!r}")
    return HYPOTHESIS_TIERS[key]


def default_cost_profile_for_universe(universe: str | None) -> str | None:
    """Universe D uses realistic Alpaca costs (tiered slippage + borrow)."""
    if str(universe or "").strip().upper() == "D":
        return "US_ALPACA_D_REALISTIC"
    return None


def star_panel_paths(
    *,
    universe: str = "D",
    bar: str = "1d",
    root: str | None = None,
) -> tuple[str, str, str]:
    """Train, full, and manifest paths for cached star-stack panels."""
    data = s2_data_dir(root)
    train = os.path.join(data, f"s2_panel_{universe}_{bar}_train_star.parquet")
    full = os.path.join(data, f"s2_panel_{universe}_{bar}_full_star.parquet")
    manifest = os.path.join(data, f"s2_panel_{universe}_{bar}_star_manifest.json")
    return train, full, manifest


def load_star_panels(
    *,
    universe: str = "D",
    bar: str = "1d",
    pair_ids: Sequence[str] | None = None,
    root: str | None = None,
    require_manifest_match: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load cached star-stack panels when manifest matches ``s2_star_stack.json``."""
    import json

    train_path, full_path, manifest_path = star_panel_paths(
        universe=universe, bar=bar, root=root
    )
    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"missing {train_path}. Run build_star_panels.ipynb after H-005/H-006 freeze."
        )
    manifest: dict = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    if require_manifest_match and os.path.isfile(DEFAULT_STAR_STACK):
        with open(DEFAULT_STAR_STACK, encoding="utf-8") as f:
            stack = json.load(f)
        if manifest.get("stack_hash") and manifest.get("stack_hash") != _stack_hash(stack):
            raise ValueError(
                "star panel manifest does not match current s2_star_stack.json; rebuild panels"
            )
    train = pd.read_parquet(train_path)
    full = pd.read_parquet(full_path) if os.path.isfile(full_path) else train.copy()
    if pair_ids is not None:
        train = filter_pairs(train, pair_ids)
        full = filter_pairs(full, pair_ids)
    train["date"] = pd.to_datetime(train["date"])
    full["date"] = pd.to_datetime(full["date"])
    return train, full, manifest


def _stack_hash(stack: dict) -> str:
    import hashlib
    import json

    payload = json.dumps(stack, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_star_panel_manifest(
    path: str,
    stack: dict,
    *,
    pair_ids: Sequence[str],
    extra: dict | None = None,
) -> None:
    """Write manifest alongside star parquets (stack hash + pair list)."""
    import json

    payload = {
        "stack_hash": _stack_hash(stack),
        "universe": stack.get("UNIVERSE_STAR"),
        "bar": stack.get("BAR_STAR"),
        "pair_ids": list(pair_ids),
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


VARIANT_LEDGER_PATH = os.path.join(ARTIFACTS_DIR, "s2_variant_ledger.json")

HYPOTHESIS_ORDER: tuple[str, ...] = (
    "H-001",
    "H-002",
    "H-003",
    "H-004",
    "H-005",
    "H-006",
    "H-007",
    "H-008",
    "H-009",
    "H-010",
    "H-011",
    "H-012",
    "H-013",
    "H-014",
    "H-015",
)


def _normalize_hyp_id(hyp_id: str) -> str:
    key = str(hyp_id).strip().upper()
    if key not in HYPOTHESIS_ORDER:
        raise KeyError(f"unknown hyp_id {hyp_id!r}; expected one of {HYPOTHESIS_ORDER}")
    return key


def load_variant_ledger(path: str | None = None) -> dict:
    p = path or VARIANT_LEDGER_PATH
    if not os.path.isfile(p):
        return {"entries": [], "cumulative_arms": 0}
    import json

    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_variant_ledger(payload: dict, path: str | None = None) -> None:
    import json

    p = path or VARIANT_LEDGER_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    entries = payload.get("entries") or []
    payload["cumulative_arms"] = int(
        sum(int(e.get("n_arms", len(e.get("arms", [])))) for e in entries)
    )
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def cumulative_trials_before(hyp_id: str, *, path: str | None = None) -> int:
    """Sum registered arms for hyps strictly before ``hyp_id`` in stack order."""
    key = _normalize_hyp_id(hyp_id)
    idx = HYPOTHESIS_ORDER.index(key)
    if idx == 0:
        return 0
    prior = set(HYPOTHESIS_ORDER[:idx])
    ledger = load_variant_ledger(path)
    total = 0
    for entry in ledger.get("entries") or []:
        eid = str(entry.get("hyp_id", "")).strip().upper()
        if eid in prior:
            total += int(entry.get("n_arms", len(entry.get("arms", []))))
    return total


def n_trials_local(configs: dict) -> int:
    return len(configs)


def n_trials_ledger_total(*, path: str | None = None) -> int:
    """Sum of all registered arms in the variant ledger (final tearsheet DSR N)."""
    ledger = load_variant_ledger(path)
    total = 0
    for entry in ledger.get("entries") or []:
        total += int(entry.get("n_arms", len(entry.get("arms", []))))
    return int(total)


def n_trials_stack(hyp_id: str, configs: dict, *, path: str | None = None) -> int:
    return cumulative_trials_before(hyp_id, path=path) + n_trials_local(configs)


def register_hypothesis_arms(
    hyp_id: str,
    arm_names: Sequence[str],
    *,
    overwrite: bool = False,
    path: str | None = None,
) -> dict:
    """Append or replace the ledger entry for a hypothesis screen."""
    key = _normalize_hyp_id(hyp_id)
    arms = [str(a) for a in arm_names]
    ledger = load_variant_ledger(path)
    entries: list[dict] = list(ledger.get("entries") or [])
    row = {"hyp_id": key, "arms": arms, "n_arms": len(arms)}
    replaced = False
    for i, entry in enumerate(entries):
        if str(entry.get("hyp_id", "")).strip().upper() == key:
            if not overwrite:
                raise ValueError(
                    f"{key} already in variant ledger; pass overwrite=True to replace"
                )
            entries[i] = row
            replaced = True
            break
    if not replaced:
        entries.append(row)
    entries.sort(key=lambda e: HYPOTHESIS_ORDER.index(_normalize_hyp_id(e["hyp_id"])))
    ledger["entries"] = entries
    save_variant_ledger(ledger, path)
    return ledger
