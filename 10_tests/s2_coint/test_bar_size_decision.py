"""Tests for H-002 bar_size_decision_table (tearsheet-backed scoreboard)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.report import bar_size_decision_table
from backtest.s2_coint.runner import S2BacktestResult
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.engine import BookSimResult


def _synthetic_result(arm: str, *, seed: int, n: int = 120) -> S2BacktestResult:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n, freq="B")
    # Slight positive drift so Sharpe / Calmar are finite.
    rets = pd.Series(rng.normal(0.0004, 0.01, size=n), index=idx, name="ret")
    i0, i1 = max(5, n // 8), max(10, n // 4)
    i2, i3 = max(15, n // 3), max(20, n // 2)
    trades = pd.DataFrame(
        {
            "pair_id": ["A|B", "A|B"],
            "side": [1, -1],
            "entry_date": [idx[i0], idx[i2]],
            "exit_date": [idx[i1], idx[i3]],
            "hold_bars": [int(i1 - i0), int(i3 - i2)],
            "entry_cost_bps": [6.6, 6.6],
            "exit_cost_bps": [6.6, 6.6],
            "exit_reason": ["z", "z"],
        }
    )
    cfg = S2SimConfig(bar="1d" if arm == "1d" else "1h")
    book = BookSimResult(returns=rets, pair_results={}, returns_base=rets)
    return S2BacktestResult(
        config=cfg,
        returns=rets,
        metrics={"ann_sharpe": float("nan"), "max_drawdown": float("nan"), "n_days": n},
        book=book,
        pair_trades=trades,
        returns_base=rets,
    )


def test_bar_size_decision_table_columns_and_arms():
    results = {
        "1d": _synthetic_result("1d", seed=1),
        "1h": _synthetic_result("1h", seed=2),
    }
    tbl = bar_size_decision_table(results, s1_weekly=None, n_trials_local=2, n_trials_stack=7)
    assert list(tbl.columns) == [
        "arm",
        "ann_sharpe_net",
        "ann_sharpe_gross",
        "max_drawdown",
        "calmar",
        "cvar_5",
        "corr_to_s1",
        "cost_bps_year",
        "n_trades",
        "n_days",
    ]
    assert set(tbl["arm"]) == {"1d", "1h"}
    # Gross should be weakly above net when costs are added back.
    for _, row in tbl.iterrows():
        assert np.isfinite(row["ann_sharpe_net"])
        assert np.isfinite(row["ann_sharpe_gross"])
        assert row["ann_sharpe_gross"] >= row["ann_sharpe_net"] - 1e-9
        assert row["n_trades"] == 2
        assert row["n_days"] == 120
        assert np.isfinite(row["cvar_5"])
        assert np.isfinite(row["max_drawdown"])
        assert row["max_drawdown"] <= 0


def test_bar_size_decision_table_empty():
    tbl = bar_size_decision_table({})
    assert tbl.empty
    assert "ann_sharpe_net" in tbl.columns


def test_plotly_bar_size_trade_compare_builds():
    from backtest.s2_coint.diagnosis import plotly_bar_size_trade_compare

    dates = pd.bdate_range("2024-01-01", periods=40, freq="B")
    panel = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "pair_id": ["A|B"] * 40 + ["C|D"] * 40,
            "z": np.linspace(-2, 2, 80),
            "spread": np.linspace(-1, 1, 80),
        }
    )
    res = _synthetic_result("1d", seed=3, n=40)
    fig = plotly_bar_size_trade_compare(
        panel, panel, res, res, pair_ids=["A|B", "C|D"], title="unit"
    )
    assert fig is not None
    assert len(fig.data) > 0
    assert fig.layout.updatemenus is not None
