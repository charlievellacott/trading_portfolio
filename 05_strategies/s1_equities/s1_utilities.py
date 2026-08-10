"""S1 strategy helpers: trade-date panel, rename, model feature cols, cleanup,
predictions cache, and PIT sizing-history inputs."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data.ingestion.equity_fetcher import OHLCV_COLUMNS, fetch_ohlcv
from data.processing.cleaner import forward_fill_panel
from models.s1_equities.training_common import (
    LABEL_COL,
    date_ic_series,
    feature_attrition_summary,
    week_start_dates,
)
from risk.position_sizing import monday_inv_vol_weights

_HLCV = ("high", "low", "close", "volume")

_HERE = os.path.dirname(os.path.abspath(__file__))
# Live cache lives under 05_strategies/s1_equities/cache/ unless S1_CACHE_DIR is set.
_DEFAULT_CACHE_DIR = os.path.join(_HERE, "cache")
_PRED_CACHE_NAME = "s1_live_predictions.parquet"
_SIZING_STATE_NAME = "s1_live_sizing_state.json"


def _cache_dir() -> str:
    override = os.environ.get("S1_CACHE_DIR", "").strip()
    if override:
        return os.path.abspath(override)
    return _DEFAULT_CACHE_DIR


# Successive week-starts are ~7 calendar days; >14 ⇒ skipped week(s) / stale hole
_MAX_WEEK_START_GAP_DAYS = 14
_PRED_CACHE_COLS = [
    "date",
    "ticker",
    "feature_date",
    "score",
    "cs_rank",
    LABEL_COL,
    "is_research_is",
]


def to_s1_trade_date_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a calendar-dated OHLCV panel to the S1 trade-date contract.

    ``date``
        Trade / fill date. Decision is made pre-open of ``date``; fill uses
        ``open`` of ``date`` (``open`` is **not** lagged).

    ``feature_date``
        Prior session for each ticker — the **info cutoff** for alt-data merges
        (FINRA / SEC / GDELT). Store attach helpers merge on ``feature_date``
        when present so same-morning information cannot leak onto the trade
        row. This column is **not** a model feature; drop it after engineering.

    Why ``high`` / ``low`` / ``close`` / ``volume`` are ``shift(1)``
        Store factor helpers read those column names as known-before-open
        inputs (momentum, beta, volume z-scores, valuation×momentum, etc.).
        Without the lag they would use same-day bars that are not known at the
        pre-open decision. Lagged OHLCV are **inputs to that math**, not lagged
        price features in the production linear model.
    """
    required = {"date", "ticker", "open", *_HLCV}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["ticker"] = work["ticker"].astype(str).str.strip().str.upper()
    work = work.sort_values(["ticker", "date"]).reset_index(drop=True)

    work["feature_date"] = work.groupby("ticker", sort=False)["date"].shift(1)
    for col in _HLCV:
        work[col] = work.groupby("ticker", sort=False)[col].shift(1)

    work = work.dropna(subset=["feature_date"]).reset_index(drop=True)
    work["feature_date"] = pd.to_datetime(work["feature_date"])

    if not work["feature_date"].lt(work["date"]).all():
        raise ValueError("feature_date must be strictly before date on every row")

    return work


def ensure_decision_date_ohlcv(
    panel: pd.DataFrame,
    decision_date: pd.Timestamp,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """
    Append dummy calendar bars for ``decision_date`` when pre-open fetch has
    no Monday row yet. Copies the last bar per ticker so ``to_s1_trade_date_panel``
    lags Friday HLC onto the Monday trade row.
    """
    dt = pd.Timestamp(decision_date).normalize()
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work["ticker"] = work["ticker"].astype(str).str.strip().str.upper()
    present = set(work.loc[work["date"] == dt, "ticker"].tolist())
    last_by_ticker = work.sort_values(["ticker", "date"]).groupby(
        "ticker", sort=False
    ).tail(1)
    rows = []
    for ticker in tickers:
        t = str(ticker).strip().upper()
        if t in present:
            continue
        prev = last_by_ticker.loc[last_by_ticker["ticker"] == t]
        if prev.empty:
            continue
        row = prev.iloc[0].to_dict()
        row["date"] = dt
        rows.append(row)
    if not rows:
        return work.sort_values(["date", "ticker"]).reset_index(drop=True)
    extra = pd.DataFrame(rows)
    extra["date"] = pd.to_datetime(extra["date"]).dt.normalize()
    out = pd.concat([work, extra], ignore_index=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def ensure_opens_decision_date(
    opens: pd.DataFrame,
    decision_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Ensure inv-vol can look up ``decision_date``. Dummy open equals the last
    session; ``pit_shift=1`` does not use today's open in the vol estimate.
    """
    dt = pd.Timestamp(decision_date).normalize()
    wide = opens.copy()
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index)).normalize()
    if dt in wide.index or wide.empty:
        return wide.sort_index()
    last = wide.iloc[-1]
    wide.loc[dt] = last.to_numpy()
    return wide.sort_index()


def rename_feature_stem(
    panel: pd.DataFrame,
    stem: str,
    target: str,
) -> pd.DataFrame:
    """
    Rename bare store stem ``stem`` to production name ``target``.

    No-op if ``target`` already present. Raises if neither ``stem`` nor
    ``target`` exists.
    """
    if target in panel.columns:
        if stem in panel.columns and stem != target:
            return panel.drop(columns=[stem])
        return panel
    if stem not in panel.columns:
        raise ValueError(
            f"expected store column {stem!r} to rename to {target!r}; "
            f"neither present"
        )
    return panel.rename(columns={stem: target})


def load_production_feature_cols(model_artifact_path: str) -> list[str]:
    """Load ``feature_cols`` from a production joblib artifact dict."""
    if not os.path.isfile(model_artifact_path):
        raise FileNotFoundError(model_artifact_path)
    obj = joblib.load(model_artifact_path)
    if not isinstance(obj, dict) or "feature_cols" not in obj:
        raise ValueError(
            f"{model_artifact_path!r} must be a dict with key 'feature_cols'"
        )
    cols = obj["feature_cols"]
    if not isinstance(cols, (list, tuple)) or not cols:
        raise ValueError("feature_cols must be a non-empty list")
    return [str(c) for c in cols]


def drop_non_model_columns(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Keep identity, OHLCV, and model features; drop raw alt workspace cols.

    Keeps ``feature_date`` when present (live predictions cache / audit).
    """
    keep = ["date", "ticker", *OHLCV_COLUMNS, *list(feature_cols)]
    if "feature_date" in panel.columns:
        keep = [
            "date",
            "ticker",
            "feature_date",
            *OHLCV_COLUMNS,
            *list(feature_cols),
        ]
    missing = [c for c in keep if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing columns after engineering: {missing}")
    out = panel.loc[:, keep].copy()
    return out


# ---------------------------------------------------------------------------
# Predictions cache paths / sizing state
# ---------------------------------------------------------------------------


def predictions_cache_path() -> str:
    """
    Absolute path to the live predictions parquet.

    Default: ``05_strategies/s1_equities/cache/s1_live_predictions.parquet``.
    Override the directory with env ``S1_CACHE_DIR``.
    """
    return os.path.join(_cache_dir(), _PRED_CACHE_NAME)


def sizing_state_path() -> str:
    """
    Absolute path to prev_leverage JSON (vol-target deadband).

    Default: ``05_strategies/s1_equities/cache/s1_live_sizing_state.json``.
    Override the directory with env ``S1_CACHE_DIR``.
    """
    return os.path.join(_cache_dir(), _SIZING_STATE_NAME)


def clear_sizing_state() -> None:
    """Drop deadband state (call when cache history is truncated across a gap)."""
    path = sizing_state_path()
    if os.path.isfile(path):
        os.remove(path)


def load_prev_leverage() -> float | None:
    path = sizing_state_path()
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    val = obj.get("prev_leverage")
    if val is None:
        return None
    return float(val)


def save_prev_leverage(prev_leverage: float) -> None:
    os.makedirs(_cache_dir(), exist_ok=True)
    with open(sizing_state_path(), "w", encoding="utf-8") as f:
        json.dump({"prev_leverage": float(prev_leverage)}, f)


def pivot_opens(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide open-price pivot: index=date, columns=ticker."""
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["ticker"] = work["ticker"].astype(str).str.strip().str.upper()
    wide = work.pivot(index="date", columns="ticker", values="open").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide.astype(float)


def fetch_opens_matrix(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch OHLCV via equity_fetcher and pivot opens."""
    frames = [fetch_ohlcv(t, start_date, end_date) for t in tickers]
    if not frames:
        raise ValueError("no OHLCV frames for opens matrix")
    panel = pd.concat(frames, ignore_index=True)
    return pivot_opens(panel)


def _attach_fwd_ret_5(panel: pd.DataFrame) -> pd.DataFrame:
    """Open-to-open 5-session forward return on a long panel (per ticker)."""
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["ticker", "date"])
    nxt = work.groupby("ticker", sort=False)["open"].shift(-5)
    work[LABEL_COL] = nxt / work["open"] - 1.0
    return work


def _score_week_start_panel(strategy: Any, panel: pd.DataFrame) -> pd.DataFrame:
    """
    Slim-ffill predict on week-start rows; return cache-shaped frame.

    Steps: ffill features → Monday filter → complete-case → score → cs_rank.
    """
    feature_cols = list(strategy.feature_cols)
    if not feature_cols:
        feature_cols = load_production_feature_cols(strategy.model_artifact_path)
        strategy.feature_cols = feature_cols

    model = strategy.load_production_model()
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["ticker"] = work["ticker"].astype(str).str.strip().str.upper()

    # 1. Unlimited within-ticker ffill (slim-ffill training parity)
    work = forward_fill_panel(work, columns=feature_cols, limit=None)

    # 2. Labels where exit open exists (daily panel, then filter)
    work = _attach_fwd_ret_5(work)

    # 3. Week-start trade dates only (mon_open_mon_open)
    weeks = week_start_dates(work["date"])
    week = work.loc[work["date"].isin(weeks)].copy()

    # 4. Complete-case features
    complete = week[feature_cols].notna().all(axis=1)
    scored = week.loc[complete].copy()
    if scored.empty:
        n_week = int(len(week))
        attrition = feature_attrition_summary(
            week, feature_cols, ffill=False
        )
        blame = attrition["total_blame"].sort_values(ascending=False)
        top_feat = list(blame.head(8).items())
        print(
            "predictions cache score: no complete week-start rows "
            f"(n_week_rows={n_week}, n_complete=0)",
            flush=True,
        )
        print("  feature NaN share (after slim-ffill, week-starts only):", flush=True)
        for name, rate in top_feat:
            print(f"    {name}: {float(rate):.3f}", flush=True)
        if n_week > 0:
            per_ticker = (
                week.assign(_ok=complete.to_numpy())
                .groupby("ticker", sort=False)["_ok"]
                .sum()
                .astype(int)
            )
            never = sorted(per_ticker.index[per_ticker == 0].astype(str).tolist())
        else:
            never = sorted(work["ticker"].astype(str).unique().tolist())
        show = never[:20]
        more = f" (+{len(never) - len(show)} more)" if len(never) > len(show) else ""
        print(
            f"  tickers with zero complete week-starts: "
            f"n={len(never)}{more}",
            flush=True,
        )
        if show:
            print(f"    {', '.join(show)}", flush=True)
        top3 = ", ".join(f"{n}={float(r):.3f}" for n, r in top_feat[:3])
        raise ValueError(
            "no complete week-start rows to score for predictions cache "
            f"(n_week_rows={n_week}, n_complete=0; top features NaN: {top3})"
        )

    # 5. Predict
    scores = np.asarray(model.predict(scored[feature_cols]), dtype=float)
    out = pd.DataFrame(
        {
            "date": scored["date"].to_numpy(),
            "ticker": scored["ticker"].to_numpy(),
            "feature_date": (
                scored["feature_date"].to_numpy()
                if "feature_date" in scored.columns
                else pd.NaT
            ),
            "score": scores,
            LABEL_COL: scored[LABEL_COL].to_numpy(),
            "is_research_is": False,
        }
    )
    out["cs_rank"] = out.groupby("date")["score"].rank(method="average", pct=True)
    out["date"] = pd.to_datetime(out["date"])
    if "feature_date" in out.columns:
        out["feature_date"] = pd.to_datetime(out["feature_date"])
    return out[_PRED_CACHE_COLS].sort_values(["date", "ticker"]).reset_index(drop=True)


def _write_predictions_cache(frame: pd.DataFrame) -> None:
    os.makedirs(_cache_dir(), exist_ok=True)
    frame.to_parquet(predictions_cache_path(), index=False)


def load_predictions_cache() -> pd.DataFrame:
    path = predictions_cache_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if "feature_date" in df.columns:
        df["feature_date"] = pd.to_datetime(df["feature_date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_predictions_cache(
    strategy: Any,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Bootstrap live predictions parquet when missing.

    1. Use ``panel`` if provided, else ``strategy.generate_features``
    2. Score week-starts with the production Ridge
    3. Persist cache under strategies/s1_equities/cache/
    """
    # 1. Feature panel (reuse caller panel to avoid a second FINRA/GDELT/SEC fetch)
    if panel is None:
        panel = strategy.generate_features()
    # 2–5. Score week-starts
    preds = _score_week_start_panel(strategy, panel)
    # 6. Write
    _write_predictions_cache(preds)
    return preds


def _week_start_index(preds: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted unique week-start dates in a predictions frame."""
    return pd.DatetimeIndex(pd.to_datetime(preds["date"].unique())).sort_values()


def _week_start_gap_breaks(
    dates: pd.DatetimeIndex,
    *,
    max_gap_days: int = _MAX_WEEK_START_GAP_DAYS,
) -> list[pd.Timestamp]:
    """
    Return each week-start that begins a new contiguous run after a gap.

    Example: dates Mon1 ... Mon4, (hole), Mon8 → returns [Mon8].
    """
    if dates is None or len(dates) < 2:
        return []
    breaks: list[pd.Timestamp] = []
    ordered = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    for a, b in zip(ordered[:-1], ordered[1:]):
        if (pd.Timestamp(b) - pd.Timestamp(a)).days > int(max_gap_days):
            breaks.append(pd.Timestamp(b))
    return breaks


def trailing_contiguous_predictions(
    preds: pd.DataFrame,
    *,
    max_gap_days: int = _MAX_WEEK_START_GAP_DAYS,
) -> pd.DataFrame:
    """
    Keep only the trailing contiguous week-start run (through the latest date).

    Drops any history that sits before a multi-week hole so rolling IC / vol
    targeting never treat a month-old row as adjacent to today.
    """
    if preds is None or preds.empty:
        return preds
    dates = _week_start_index(preds)
    breaks = _week_start_gap_breaks(dates, max_gap_days=max_gap_days)
    if not breaks:
        return preds.sort_values(["date", "ticker"]).reset_index(drop=True)
    # Last break is the start of the newest contiguous segment
    cut = breaks[-1]
    out = preds.loc[pd.to_datetime(preds["date"]) >= cut].copy()
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def ensure_predictions_cache_continuity(
    cached: pd.DataFrame,
    strategy: Any,
    panel: pd.DataFrame | None,
    *,
    max_gap_days: int = _MAX_WEEK_START_GAP_DAYS,
) -> pd.DataFrame:
    """
    Enforce contiguous week-starts in the live predictions cache.

    1. If gaps exist and a feature panel is available, re-score it and merge
       (fills Mondays that were skipped when the strategy was not run).
    2. If gaps still remain, keep only the trailing contiguous run and clear
       ``prev_leverage`` (deadband state is invalid across a hole).
    """
    dates = _week_start_index(cached)
    if not _week_start_gap_breaks(dates, max_gap_days=max_gap_days):
        return cached

    # 1. Try to fill holes by re-scoring the full panel
    if panel is not None and not panel.empty:
        filled = _score_week_start_panel(strategy, panel)
        if not filled.empty:
            keys = ["date", "ticker"]
            old_idx = pd.MultiIndex.from_frame(cached[keys])
            new_idx = pd.MultiIndex.from_frame(filled[keys])
            keep = ~old_idx.isin(new_idx)
            cached = pd.concat(
                [cached.loc[keep], filled], ignore_index=True
            )
            cached = (
                cached[_PRED_CACHE_COLS]
                .sort_values(["date", "ticker"])
                .reset_index(drop=True)
            )
            dates = _week_start_index(cached)
            if not _week_start_gap_breaks(dates, max_gap_days=max_gap_days):
                return cached

    # 2. Still gapped → trailing contiguous only; reset leverage deadband
    trimmed = trailing_contiguous_predictions(cached, max_gap_days=max_gap_days)
    clear_sizing_state()
    return trimmed


def _backfill_fwd_ret_from_opens(preds: pd.DataFrame, opens: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN fwd_ret_5 where open[d+5] now exists on the open pivot."""
    out = preds.copy()
    if out.empty or opens.empty:
        return out
    opens = opens.copy()
    opens.index = pd.to_datetime(opens.index)
    cal = opens.index.sort_values()
    # Map each calendar date → open 5 sessions later (same column set)
    pos = {d: i for i, d in enumerate(cal)}
    rows = []
    for i, row in out.iterrows():
        if pd.notna(row.get(LABEL_COL)):
            rows.append(row[LABEL_COL])
            continue
        dt = pd.Timestamp(row["date"])
        ticker = str(row["ticker"]).upper()
        if dt not in pos or ticker not in opens.columns:
            rows.append(np.nan)
            continue
        j = pos[dt] + 5
        if j >= len(cal):
            rows.append(np.nan)
            continue
        o0 = opens.at[dt, ticker]
        o1 = opens.at[cal[j], ticker]
        if pd.isna(o0) or pd.isna(o1) or float(o0) == 0.0:
            rows.append(np.nan)
        else:
            rows.append(float(o1) / float(o0) - 1.0)
    out[LABEL_COL] = rows
    return out


def update_predictions_cache(
    strategy: Any,
    panel_with_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Daily cache update: create if missing, else upsert scored rows + backfill labels.

    ``panel_with_scores`` may be a feature panel (no scores yet) or already scored;
    if ``prediction``/``score`` missing, scores are built via the production model.
    """
    path = predictions_cache_path()
    # 1. Bootstrap if absent
    if not os.path.isfile(path):
        return build_predictions_cache(strategy, panel=panel_with_scores)

    # 2. Load existing
    cached = load_predictions_cache()

    # 3. Build / accept new week-start score rows
    if panel_with_scores is None:
        panel_with_scores = strategy.generate_features()
    if (
        "score" not in panel_with_scores.columns
        and "prediction" not in panel_with_scores.columns
    ):
        new_rows = _score_week_start_panel(strategy, panel_with_scores)
    else:
        work = panel_with_scores.copy()
        if "score" not in work.columns:
            work = work.rename(columns={"prediction": "score"})
        work["date"] = pd.to_datetime(work["date"])
        weeks = week_start_dates(work["date"])
        work = work.loc[work["date"].isin(weeks)].copy()
        if LABEL_COL not in work.columns:
            work = _attach_fwd_ret_5(work)
        if "cs_rank" not in work.columns:
            work["cs_rank"] = work.groupby("date")["score"].rank(
                method="average", pct=True
            )
        if "is_research_is" not in work.columns:
            work["is_research_is"] = False
        if "feature_date" not in work.columns:
            work["feature_date"] = pd.NaT
        new_rows = work[_PRED_CACHE_COLS].copy()

    # 4. Upsert by (date, ticker) — new rows replace overlapping keys
    if not new_rows.empty:
        keys = ["date", "ticker"]
        old_idx = pd.MultiIndex.from_frame(cached[keys])
        new_idx = pd.MultiIndex.from_frame(new_rows[keys])
        keep = ~old_idx.isin(new_idx)
        cached = pd.concat([cached.loc[keep], new_rows], ignore_index=True)

    # 5. Backfill labels from opens covering the cache span
    tickers = sorted(cached["ticker"].astype(str).unique())
    start = str(pd.Timestamp(cached["date"].min()).date())
    end = str(pd.Timestamp(strategy.end_date).date())
    opens = fetch_opens_matrix(tickers, start, end)
    cached = _backfill_fwd_ret_from_opens(cached, opens)

    cached = (
        cached[_PRED_CACHE_COLS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    # 6. Contiguous week-starts only (fill holes from panel, else truncate)
    #    Prevents a month-later run from sitting next to stale rows for IC / vol.
    panel_for_fill = panel_with_scores
    if panel_for_fill is not None and (
        "score" in panel_for_fill.columns or "prediction" in panel_for_fill.columns
    ):
        # Prefer raw feature panel for re-score fill; drop score cols if present
        drop_cols = [c for c in ("score", "prediction", "cs_rank") if c in panel_for_fill.columns]
        if drop_cols:
            # Still usable as feature panel if model feature cols remain
            panel_for_fill = panel_for_fill.drop(columns=drop_cols, errors="ignore")
    cached = ensure_predictions_cache_continuity(
        cached, strategy, panel_for_fill
    )

    # 7. Persist
    _write_predictions_cache(cached)
    return cached


def load_or_update_predictions_cache(
    strategy: Any,
    panel_with_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Single entrypoint: ensure cache exists and is current."""
    return update_predictions_cache(strategy, panel_with_scores)


# ---------------------------------------------------------------------------
# PIT history inputs for monday_gross_leverage (not weight orchestration)
# ---------------------------------------------------------------------------


def pit_safe_ic_inputs(
    preds: pd.DataFrame,
    decision_date: pd.Timestamp,
    *,
    pit_lag: int = 1,
) -> pd.DataFrame:
    """
    IC + n_names for weeks whose labels are known before ``decision_date``.

    Mon→Mon ``pit_lag=1`` drops the most recent week-start that would exit at
    this Monday open. History is restricted to a trailing contiguous week-start
    run first so a cache hole cannot enter the Bayesian IC update.
    """
    # 0. Contiguous week-starts only
    preds = trailing_contiguous_predictions(preds)

    # 1. Per-date Spearman IC vs fwd_ret_5
    ics = date_ic_series(preds, "score", label_col=LABEL_COL)
    n_names = preds.groupby("date")["ticker"].nunique()
    n_names.index = pd.to_datetime(n_names.index)
    frame = pd.DataFrame({"ic": ics, "n_names": n_names}).sort_index()
    frame = frame.dropna(subset=["ic"])

    # 2. Only entry dates strictly before decision_date
    dt = pd.Timestamp(decision_date)
    usable = frame.loc[frame.index < dt]
    # 3. Exclude last ``pit_lag`` completed weeks (exit at this open)
    if pit_lag > 0 and len(usable) >= pit_lag:
        usable = usable.iloc[:-pit_lag]
    elif pit_lag > 0:
        usable = usable.iloc[0:0]
    return usable


def base_period_returns_from_cache(
    preds: pd.DataFrame,
    opens: pd.DataFrame,
    *,
    decision_date: pd.Timestamp,
    n: int,
    inv_vol_window: int,
    pit_lag: int = 1,
) -> pd.Series:
    """
    Unlevered gross open-to-open book returns for completed Mon→Mon holds.

    Rebuilds inv-vol base weights per historical week-start via
    ``monday_inv_vol_weights``. Costs/stops omitted (live sizing ledger).
    Uses only a trailing contiguous week-start run from the cache.
    """
    # 0. Contiguous week-starts only
    preds = trailing_contiguous_predictions(preds)

    # 1. Full week-start calendar from cache; usable entries before decision
    full = pd.DatetimeIndex(pd.to_datetime(preds["date"].unique())).sort_values()
    dt = pd.Timestamp(decision_date)
    usable = full[full < dt]
    if pit_lag > 0 and len(usable) >= pit_lag:
        usable = usable[:-pit_lag]
    elif pit_lag > 0:
        return pd.Series(dtype=float)

    opens = opens.copy()
    opens.index = pd.to_datetime(opens.index)
    rets: dict[pd.Timestamp, float] = {}

    # 2. Each entry → next week-start exit on the full calendar
    for entry in usable:
        nxt = full[full > entry]
        if nxt.empty:
            continue
        exit_ = nxt[0]
        if entry not in opens.index or exit_ not in opens.index:
            continue

        day = preds.loc[pd.to_datetime(preds["date"]) == entry, ["ticker", "score"]]
        if day.empty:
            continue
        scores = day.set_index("ticker")["score"]
        w = monday_inv_vol_weights(
            scores,
            opens,
            decision_date=entry,
            n=n,
            window=inv_vol_window,
        )
        if w.empty:
            continue

        # 3. Book return = sum_i w_i * (open_exit / open_entry - 1)
        o0 = opens.loc[entry].reindex(w.index).astype(float)
        o1 = opens.loc[exit_].reindex(w.index).astype(float)
        asset_r = o1 / o0 - 1.0
        valid = w.index[np.isfinite(asset_r.to_numpy()) & np.isfinite(w.to_numpy())]
        if len(valid) == 0:
            continue
        rets[pd.Timestamp(entry)] = float(
            (w.loc[valid] * asset_r.loc[valid]).sum()
        )

    return pd.Series(rets, dtype=float).sort_index()
