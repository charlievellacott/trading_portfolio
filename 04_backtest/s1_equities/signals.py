"""Load S1 prediction scores and price panels for backtests."""

from __future__ import annotations

import os

import pandas as pd

from models.s1_equities.training_common import default_paths

TIMING_MON_OPEN_MON_OPEN = "mon_open_mon_open"
TIMING_MON_OPEN_FRI_CLOSE = "mon_open_fri_close"
VALID_TIMING_MODES = (TIMING_MON_OPEN_MON_OPEN, TIMING_MON_OPEN_FRI_CLOSE)

DEFAULT_PRED_NAME = "s1_linear_slim_ffill_is_predictions.parquet"
DEFAULT_PRED_NAME_MON_FRI = "s1_linear_slim_ffill_mon_fri_is_predictions.parquet"


def default_backtest_paths(root: str) -> dict[str, str]:
    """Artifact / data paths for S1 backtests."""
    model_paths = default_paths(root)
    bt_root = os.path.join(root, "04_backtest", "s1_equities")
    extras_dir = os.path.join(
        root, "03_models", "s1_equities", "model_tests", "extras"
    )
    return {
        **model_paths,
        "predictions": os.path.join(model_paths["model_dir"], DEFAULT_PRED_NAME),
        "predictions_mon_fri": os.path.join(extras_dir, DEFAULT_PRED_NAME_MON_FRI),
        "bt_root": bt_root,
        "artifacts": os.path.join(bt_root, "artifacts"),
        "notebooks": os.path.join(bt_root, "notebooks"),
    }


def prediction_path_for_timing(paths: dict[str, str], timing_mode: str) -> str:
    """
    Prediction parquet matched to hold interval / training label.

    - ``mon_open_mon_open`` → slim-ffill Ridge on ``fwd_ret_5``
    - ``mon_open_fri_close`` → extras slim-ffill Ridge on ``fwd_ret_mon_fri``
    """
    if timing_mode == TIMING_MON_OPEN_FRI_CLOSE:
        return paths["predictions_mon_fri"]
    if timing_mode == TIMING_MON_OPEN_MON_OPEN:
        return paths["predictions"]
    raise ValueError(f"unknown timing_mode={timing_mode!r}")


def load_predictions(path: str) -> pd.DataFrame:
    """Load prediction parquet; normalize dtypes."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if "feature_date" in df.columns:
        df["feature_date"] = pd.to_datetime(df["feature_date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    if "score" not in df.columns:
        raise ValueError("predictions missing 'score'")
    if "is_research_is" not in df.columns:
        raise ValueError("predictions missing 'is_research_is'")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def score_matrix(preds: pd.DataFrame) -> pd.DataFrame:
    """Wide score panel: index=date, columns=ticker."""
    wide = preds.pivot(index="date", columns="ticker", values="score").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def is_mask_from_preds(preds: pd.DataFrame) -> pd.Series:
    """Boolean Series indexed by unique dates: True on research IS week-starts."""
    g = (
        preds.groupby("date", sort=True)["is_research_is"]
        .any()
        .sort_index()
    )
    g.index = pd.to_datetime(g.index)
    return g.astype(bool)


def load_ohlc_panels(
    features_path: str,
    *,
    tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Daily OHLC pivots from the engineered feature matrix.

    Reads ``date``, ``ticker``, ``open``, ``high``, ``low``, ``close`` from
    ``features_path`` (S1 engineered panel parquet).

    Returns
    -------
    opens, highs, lows, closes : DataFrames indexed by date, columns ticker
    """
    if not os.path.isfile(features_path):
        raise FileNotFoundError(features_path)
    cols = ["date", "ticker", "open", "high", "low", "close"]
    df = pd.read_parquet(features_path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    if tickers is not None:
        keep = {t.upper() for t in tickers}
        df = df.loc[df["ticker"].isin(keep)]
    opens = df.pivot(index="date", columns="ticker", values="open").sort_index()
    highs = df.pivot(index="date", columns="ticker", values="high").sort_index()
    lows = df.pivot(index="date", columns="ticker", values="low").sort_index()
    closes = df.pivot(index="date", columns="ticker", values="close").sort_index()
    for frame in (opens, highs, lows, closes):
        frame.index = pd.to_datetime(frame.index)
    return opens, highs, lows, closes
