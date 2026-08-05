"""Shared helpers for S1 equities model-training notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.processing.cleaner import forward_fill_panel
from data.processing.feature_implementation.utilities import cross_sectional_pct_rank

KEY_COLS = ["date", "ticker", "feature_date"]
LABEL_COL = "fwd_ret_5"
TARGET_COL = "fwd_ret_5_cs_pct"
LABEL_COL_MON_FRI = "fwd_ret_mon_fri"
TARGET_COL_MON_FRI = "fwd_ret_mon_fri_cs_pct"
RELEVANCE_COL = "relevance"
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
FWD_RET_COLS = ["fwd_ret_1", "fwd_ret_5", "fwd_ret_21", LABEL_COL_MON_FRI]
FFILL_FACTORS = ["gross_profitability", "earnings_yield"]

EXCLUDE_FROM_FEATURES = set(KEY_COLS) | set(OHLCV_COLS) | set(FWD_RET_COLS) | {
    TARGET_COL,
    TARGET_COL_MON_FRI,
    RELEVANCE_COL,
    "is_research_is",
    "is_week_start",
    "score",
    "cs_rank",
}


def default_paths(root: str) -> dict[str, str]:
    """Standard S1 engineered-feature / panel / model artifact paths."""
    data_dir = os.path.join(root, "01_data", "data_files", "s1_equities")
    model_dir = os.path.join(root, "03_models", "s1_equities")
    return {
        "features": os.path.join(data_dir, "s1_engineered_features.parquet"),
        "train_panel": os.path.join(data_dir, "s1_factor_panel_train.parquet"),
        "model_dir": os.path.join(model_dir, "model_artifacts"),
        "tearsheet_dir": os.path.join(model_dir, "model_tests", "tearsheets"),
    }


def week_start_dates(dates: pd.Series) -> pd.DatetimeIndex:
    """First trading day present per ISO year-week."""
    d = pd.DatetimeIndex(pd.to_datetime(dates).drop_duplicates()).sort_values()
    iso = d.isocalendar()
    ws = (
        pd.DataFrame({"date": d.to_numpy(), "yw": list(zip(iso.year, iso.week))})
        .groupby("yw", sort=False)["date"]
        .min()
    )
    return pd.DatetimeIndex(pd.to_datetime(ws.to_numpy())).sort_values()


def _iso_year_week(ts: pd.Timestamp) -> tuple[int, int]:
    iso = pd.Timestamp(ts).isocalendar()
    return int(iso.year), int(iso.week)


def friday_close_dates(
    trading_calendar: pd.DatetimeIndex,
) -> dict[pd.Timestamp, pd.Timestamp]:
    """
    Map each calendar session to the Fri-close exit of its ISO week.

    Exit = last trading day in the same ISO week with weekday <= Friday
    (prefer Friday when present). Same rule as S1 ``mon_open_fri_close``.
    """
    cal = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).sort_values().unique()
    by_week: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for d in cal:
        key = _iso_year_week(d)
        if pd.Timestamp(d).weekday() > 4:
            continue
        by_week.setdefault(key, []).append(pd.Timestamp(d))

    week_exit: dict[tuple[int, int], pd.Timestamp] = {}
    for key, days in by_week.items():
        fridays = [x for x in days if x.weekday() == 4]
        week_exit[key] = fridays[-1] if fridays else days[-1]

    return {pd.Timestamp(d): week_exit[_iso_year_week(d)] for d in cal if _iso_year_week(d) in week_exit}


def add_fwd_ret_mon_fri(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ``fwd_ret_mon_fri`` = Friday close / entry open - 1.

    S1 trade-date panels store same-day ``open`` on ``date`` but lag ``close``
    onto ``feature_date``. Calendar close for day ``t`` is therefore the panel
    ``close`` on rows where ``feature_date == t``.
    """
    required = {"date", "ticker", "feature_date", "open", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns for mon-fri label: {sorted(missing)}")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["feature_date"] = pd.to_datetime(out["feature_date"])

    cal_close = (
        out.dropna(subset=["feature_date"])
        .groupby(["feature_date", "ticker"], sort=False)["close"]
        .first()
        .rename("cal_close")
    )
    cal_close.index = cal_close.index.set_names(["exit_date", "ticker"])

    cal = pd.DatetimeIndex(out["date"].dropna().unique()).sort_values()
    exit_map = friday_close_dates(cal)
    out["exit_date"] = out["date"].map(exit_map)

    merged = out.merge(
        cal_close.reset_index(),
        on=["exit_date", "ticker"],
        how="left",
    )
    o = merged["open"].astype(float)
    c = merged["cal_close"].astype(float)
    ok = o.notna() & c.notna() & np.isfinite(o) & np.isfinite(c) & (o > 0) & (c > 0)
    merged[LABEL_COL_MON_FRI] = np.where(ok, c / o - 1.0, np.nan)
    return merged.drop(columns=["exit_date", "cal_close"])


def load_engineered_features(features_path: str) -> pd.DataFrame:
    """Load engineered feature matrix; sort by date, ticker."""
    if not os.path.isfile(features_path):
        raise FileNotFoundError(
            f"Missing {features_path}. Run s1_feature_matrix.ipynb export first."
        )
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    if "feature_date" in df.columns:
        df["feature_date"] = pd.to_datetime(df["feature_date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_columns(
    frame: pd.DataFrame,
    *,
    exclude: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Numeric columns eligible as model features."""
    ban = set(EXCLUDE_FROM_FEATURES if exclude is None else exclude)
    return [
        c
        for c in frame.columns
        if c not in ban and pd.api.types.is_numeric_dtype(frame[c])
    ]


def prepare_s1_week_panel(
    features_path: str,
    train_panel_path: str,
    *,
    ffill_cols: list[str] | None = None,
    exclude_from_features: set[str] | frozenset[str] | None = None,
    attach_mon_fri_label: bool = False,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Load engineered features, optional ffill, week-start filter, IS mask.

    Parameters
    ----------
    attach_mon_fri_label :
        If True, attach ``fwd_ret_mon_fri`` (Mon open → Fri close) before the
        week-start filter.

    Returns
    -------
    week_panel : week-start rows with ``is_research_is``
    feature_cols : numeric feature column names
    prices : daily open pivot (full calendar before week filter), for Alphalens
    """
    df = load_engineered_features(features_path)
    missing_req = [c for c in [LABEL_COL, "open"] if c not in df.columns]
    if missing_req:
        raise ValueError(f"engineered features missing {missing_req}")

    if attach_mon_fri_label:
        df = add_fwd_ret_mon_fri(df)

    cols = list(FFILL_FACTORS if ffill_cols is None else ffill_cols)
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = forward_fill_panel(df, columns=cols, limit=None)

    prices = df.pivot(index="date", columns="ticker", values="open").sort_index()
    prices.index = pd.to_datetime(prices.index)

    week_starts = week_start_dates(df["date"])
    df = df.loc[df["date"].isin(week_starts)].copy()

    if not os.path.isfile(train_panel_path):
        raise FileNotFoundError(train_panel_path)
    train_panel_dates = pd.Index(
        pd.to_datetime(pd.read_parquet(train_panel_path, columns=["date"])["date"])
        .dropna()
        .unique()
    ).sort_values()
    df["is_research_is"] = df["date"].isin(train_panel_dates)

    feat_cols = feature_columns(df, exclude=exclude_from_features)
    return df, feat_cols, prices


def add_cs_pct_target(
    frame: pd.DataFrame,
    *,
    label_col: str = LABEL_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Cross-sectional pct rank of ``label_col`` within each date."""
    out = frame.copy()
    out[target_col] = cross_sectional_pct_rank(out, label_col, by="date")
    return out


def add_relevance_quintiles(
    frame: pd.DataFrame,
    *,
    label_col: str = LABEL_COL,
    relevance_col: str = RELEVANCE_COL,
) -> pd.DataFrame:
    """Integer 0..4 relevance = CS quintile of forward return within date (4 = best)."""
    out = frame.copy()

    def _q(s: pd.Series) -> pd.Series:
        ok = s.notna()
        ranks = pd.Series(np.nan, index=s.index, dtype=float)
        if int(ok.sum()) < 5:
            return ranks
        r = s.loc[ok].rank(method="first")
        try:
            buckets = pd.qcut(r, 5, labels=False, duplicates="drop")
        except ValueError:
            buckets = pd.Series(0, index=r.index)
        ranks.loc[ok] = buckets.astype(float)
        return ranks

    out[relevance_col] = out.groupby("date", sort=False)[label_col].transform(_q)
    return out


def drop_nonfinite_labels(
    frame: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Keep rows with finite values in ``cols``; preserve feature NaNs."""
    mask = np.ones(len(frame), dtype=bool)
    for c in cols:
        mask &= np.isfinite(frame[c].to_numpy(dtype=float, copy=False))
    return frame.loc[mask].sort_values(["date", "ticker"]).reset_index(drop=True)


@dataclass(frozen=True)
class ChronologicalSplit:
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    embargo_dates: pd.DatetimeIndex
    is_end: pd.Timestamp
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    holdout_df: pd.DataFrame


def chronological_is_split(
    frame: pd.DataFrame,
    *,
    val_frac: float = 0.15,
    embargo_weeks: int = 1,
) -> ChronologicalSplit:
    """
    Within research IS week-starts: train / embargo / val; post-IS holdout.

    Holdout = ``~is_research_is & date > is_end`` (pre-IS padding excluded).
    """
    is_dates = pd.DatetimeIndex(
        frame.loc[frame["is_research_is"], "date"].drop_duplicates().sort_values()
    )
    n_is = len(is_dates)
    n_val = max(int(round(n_is * val_frac)), 1)
    if n_val + embargo_weeks + 20 >= n_is:
        raise ValueError(
            f"IS week-starts={n_is} too short for val_frac={val_frac} "
            f"+ embargo_weeks={embargo_weeks}"
        )

    val_dates = is_dates[-n_val:]
    train_dates = is_dates[: -(n_val + embargo_weeks)]
    embargo_dates = is_dates[len(train_dates) : len(train_dates) + embargo_weeks]
    is_end = is_dates.max()
    holdout_mask = (~frame["is_research_is"]) & (frame["date"] > is_end)

    train_df = frame.loc[frame["date"].isin(train_dates)].sort_values(
        ["date", "ticker"]
    )
    val_df = frame.loc[frame["date"].isin(val_dates)].sort_values(["date", "ticker"])
    holdout_df = frame.loc[holdout_mask].sort_values(["date", "ticker"])

    return ChronologicalSplit(
        train_dates=train_dates,
        val_dates=val_dates,
        embargo_dates=embargo_dates,
        is_end=is_end,
        train_df=train_df,
        val_df=val_df,
        holdout_df=holdout_df,
    )


def group_sizes(frame: pd.DataFrame) -> list[int]:
    """Per-date row counts for LightGBM ranking ``group``."""
    return frame.groupby("date", sort=True).size().tolist()


def date_ic_series(
    frame: pd.DataFrame,
    score_col: str = "score",
    *,
    label_col: str = LABEL_COL,
) -> pd.Series:
    """Per-date Spearman IC of ``score_col`` vs ``label_col``."""
    rows: list[tuple[object, float]] = []
    for dt, g in frame.groupby("date", sort=True):
        if len(g) < 5 or g[score_col].nunique() < 2 or g[label_col].nunique() < 2:
            rows.append((dt, np.nan))
            continue
        rho = spearmanr(g[score_col], g[label_col]).correlation
        rows.append((dt, float(rho) if rho == rho else np.nan))
    return pd.Series({d: v for d, v in rows}).sort_index()


def mean_date_ic(
    frame: pd.DataFrame,
    score_col: str = "score",
    *,
    label_col: str = LABEL_COL,
) -> dict[str, float | int]:
    """Mean / ICIR of per-date Spearman IC vs ``label_col``."""
    ics = date_ic_series(frame, score_col, label_col=label_col).dropna()
    if ics.empty:
        return {"mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan, "n": 0}
    mu, sd = float(ics.mean()), float(ics.std(ddof=1))
    return {
        "mean_ic": mu,
        "std_ic": sd,
        "icir": mu / sd if sd > 1e-12 else np.nan,
        "n": int(len(ics)),
    }


def ic_segment_table(
    preds: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    is_end: pd.Timestamp,
    *,
    score_col: str = "score",
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """
    Four-row IC summary: IS train / IS val / IS train+val / OS holdout.

    Holdout = ``~is_research_is & date > is_end``.
    """
    is_train = preds.loc[preds["date"].isin(train_dates)]
    is_val = preds.loc[preds["date"].isin(val_dates)]
    is_tv = preds.loc[preds["date"].isin(train_dates.union(val_dates))]
    holdout = preds.loc[(~preds["is_research_is"]) & (preds["date"] > is_end)]

    rows = []
    for name, part in (
        ("IS train", is_train),
        ("IS val", is_val),
        ("IS train+val", is_tv),
        ("OS holdout", holdout),
    ):
        sm = mean_date_ic(part, score_col, label_col=label_col)
        rows.append(
            {
                "segment": name,
                "mean_ic": sm["mean_ic"],
                "icir": sm["icir"],
                "n_dates": sm["n"],
            }
        )
    return pd.DataFrame(rows).set_index("segment")


def cs_rank_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    fill_value: float | None = 0.5,
) -> pd.DataFrame:
    """
    Within-date CS pct-rank each feature.

    If ``fill_value`` is not None, missing ranks are filled with that value
    (default 0.5 = neutral). Pass ``fill_value=None`` to leave NaNs.
    """
    out = pd.DataFrame(index=frame.index)
    for c in feature_cols:
        ranked = cross_sectional_pct_rank(frame, c, by="date")
        out[c] = ranked if fill_value is None else ranked.fillna(fill_value)
    return out


def attach_scores(
    frame: pd.DataFrame,
    scores: np.ndarray | pd.Series,
    *,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Build prediction frame with ``score`` and within-date ``cs_rank``."""
    out = frame[["date", "ticker", label_col, "is_research_is"]].copy()
    out["score"] = np.asarray(scores, dtype=float)
    out["cs_rank"] = out.groupby("date")["score"].rank(method="average", pct=True)
    return out
