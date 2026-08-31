"""Load sealed OOS period returns and align SPY (EV package only)."""

from __future__ import annotations

import os

import pandas as pd

from data.ingestion.equity_fetcher import fetch_ohlcv
from strategies.s2_coint.metrics import compound_to_s1_weeks, load_s1_period_returns

S1_EXPORT_NOTE = (
    "Export from 04_backtest/s1_equities/notebooks/08_oos_tearsheet.ipynb"
)
S2_EXPORT_NOTE = (
    "Export from 04_backtest/s2_coint/notebooks/01_star_tearsheet.ipynb"
)


def find_repo_root(start: str | None = None) -> str:
    """Walk parents until ``pyproject.toml`` + ``06_risk`` are found."""
    cur = os.path.abspath(start or os.getcwd())
    for _ in range(12):
        if os.path.isfile(os.path.join(cur, "pyproject.toml")) and os.path.isdir(
            os.path.join(cur, "06_risk")
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(
        "could not locate repo root (pyproject.toml + 06_risk) from "
        + os.path.abspath(start or os.getcwd())
    )


def s1_period_returns_path(repo_root: str) -> str:
    return os.path.join(
        repo_root, "01_data", "data_files", "s1_equities", "s1_period_returns.parquet"
    )


def s2_period_returns_path(repo_root: str) -> str:
    return os.path.join(
        repo_root, "01_data", "data_files", "s2_coint", "s2_period_returns.parquet"
    )


def require_parquet(path: str, hint: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sealed period returns missing: {path}. {hint}")


def load_s2_period_returns(path: str) -> pd.Series:
    """Daily S2 net book returns (date index, column ``ret``). Missing → error."""
    require_parquet(path, S2_EXPORT_NOTE)
    df = pd.read_parquet(path)
    if "ret" in df.columns:
        s = df["ret"]
    else:
        s = df.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index)
    out = s.dropna().sort_index().astype(float)
    out.name = "s2"
    return out


def load_sealed_s1(repo_root: str) -> pd.Series:
    path = s1_period_returns_path(repo_root)
    require_parquet(path, S1_EXPORT_NOTE)
    s = load_s1_period_returns(path)
    if s.empty:
        raise FileNotFoundError(
            f"Sealed S1 period returns are empty: {path}. {S1_EXPORT_NOTE}"
        )
    s.name = "strategy"
    return s


def load_sealed_s2(repo_root: str) -> pd.Series:
    s = load_s2_period_returns(s2_period_returns_path(repo_root))
    s.name = "strategy"
    return s


def spy_daily_returns(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.Series:
    """SPY close-to-close daily simple returns via ``fetch_ohlcv``."""
    start_s = (pd.Timestamp(start) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end_s = (pd.Timestamp(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    spy = fetch_ohlcv("SPY", start_s, end_s, auto_adjust=True)
    if spy is None or spy.empty:
        return pd.Series(dtype=float, name="spy")
    px = spy.set_index(pd.to_datetime(spy["date"]))["close"].astype(float).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    out = px.pct_change(fill_method=None).dropna()
    out.name = "spy"
    return out


def aligned_strategy_spy(
    strategy: pd.Series,
    *,
    bar: str,
) -> pd.DataFrame:
    """Inner-join strategy with SPY on the strategy's own bar frequency.

    ``bar='W'``: compound SPY daily to the S1 Monday–Monday week index.
    ``bar='D'``: align on overlapping daily dates.
    Independent resampling of the two series is invalid for P(beat SPY).
    """
    s = pd.to_numeric(strategy, errors="coerce").astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    if s.empty:
        raise ValueError("strategy returns are empty")
    spy = spy_daily_returns(s.index.min(), s.index.max())
    if spy.empty:
        raise RuntimeError("SPY daily returns are empty (fetch_ohlcv failed)")
    if bar.upper().startswith("W"):
        spy_b = compound_to_s1_weeks(spy, pd.DatetimeIndex(s.index))
        spy_b.name = "spy"
    elif bar.upper().startswith("D"):
        spy_b = spy.reindex(s.index)
    else:
        raise ValueError("bar must be 'D' or 'W'")
    s = s.rename("strategy")
    frame = pd.concat([s, spy_b.rename("spy")], axis=1).dropna(how="any")
    if frame.empty:
        raise ValueError("no overlapping dates between strategy and SPY")
    return frame.sort_index()
