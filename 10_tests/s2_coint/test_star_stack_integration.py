"""S2 still works after star-stack I/O moved to backtest.star_stack_io."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.research import DEFAULT_STAR_STACK, config_from_stack
from backtest.star_stack_io import load_star_stack
from execution.s2_coint.s2_paper_runner import planned_orders
from strategies.s2_coint.live_decision import walk_live_book


_REQUIRED = (
    "UNIVERSE_STAR",
    "BAR_STAR",
    "PAIRS_STAR",
    "SIZE_STAR",
    "VOL_STAR",
    "VT_TARGET_ANN_VOL_STAR",
    "OVERLAP_STAR",
)


def _oscillating_book(n: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(0)
    rows = []
    pairs = (("AAA", "BBB"), ("CCC", "DDD"), ("EEE", "FFF"))
    for pid, (ty, tx) in ((f"{a}|{b}", (a, b)) for a, b in pairs):
        shock_y = rng.normal(0.0, 1.2, n).cumsum()
        shock_x = rng.normal(0.0, 0.6, n).cumsum()
        for i, dt in enumerate(dates):
            z = float(np.sin(i / 4.0) * 3.0)
            py = 100.0 + i * 0.05 + float(shock_y[i])
            px = 50.0 + i * 0.025 + float(shock_x[i])
            rows.append(
                {
                    "date": dt,
                    "pair_id": pid,
                    "ticker_y": ty,
                    "ticker_x": tx,
                    "open_y": py,
                    "high_y": py + 0.5,
                    "low_y": py - 0.5,
                    "close_y": py,
                    "open_x": px,
                    "high_x": px + 0.25,
                    "low_x": px - 0.25,
                    "close_x": px,
                    "beta": 1.0,
                    "z": z,
                    "adf_pvalue": 0.01,
                    "half_life": 10.0,
                    "spread": z,
                    "variance_jump": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_committed_s2_stack_has_required_keys():
    stack = load_star_stack(DEFAULT_STAR_STACK)
    for key in _REQUIRED:
        assert key in stack, key
    assert len(stack["PAIRS_STAR"]) == 3
    assert stack["VT_TARGET_ANN_VOL_STAR"] == pytest.approx(0.06)


def test_config_from_stack_reads_vt_target():
    stack = load_star_stack(DEFAULT_STAR_STACK)
    cfg = config_from_stack(stack)
    assert cfg.vt_target_ann_vol == pytest.approx(0.06)
    assert cfg.vol_mode == "s1_vt"
    assert cfg.size_mode == "score"
    assert cfg.overlap_mode == "never_allow"


def test_report_reexports_same_load_star_stack():
    from backtest.s2_coint import report as s2_report
    from backtest import star_stack_io as shared

    assert s2_report.load_star_stack is shared.load_star_stack
    assert s2_report.save_star_stack is shared.save_star_stack
    assert s2_report.require_star is shared.require_star


def test_walk_live_book_vt_target_changes_leverage():
    stack = load_star_stack(DEFAULT_STAR_STACK)
    panel = _oscillating_book(90)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    cfg_lo = config_from_stack(
        stack, z_window=10, ols_window=20, adf_window=20, sigma_window=20
    )
    cfg_hi = config_from_stack(
        stack,
        z_window=10,
        ols_window=20,
        adf_window=20,
        sigma_window=20,
        vt_target_ann_vol=0.18,
    )
    book_lo = walk_live_book(panel, cfg_lo, universe_tickers=tickers)
    book_hi = walk_live_book(panel, cfg_hi, universe_tickers=tickers)
    assert "leverage" in book_lo.weights.columns
    assert book_lo.leverage > 0
    assert book_hi.leverage > book_lo.leverage


def test_planned_orders_from_live_weights():
    stack = load_star_stack(DEFAULT_STAR_STACK)
    panel = _oscillating_book(90)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    cfg = config_from_stack(
        stack, z_window=10, ols_window=20, adf_window=20, sigma_window=20
    )
    book = walk_live_book(panel, cfg, universe_tickers=tickers)
    plans = planned_orders(
        book.weights,
        equity=100_000.0,
        current={},
        universe=tickers,
    )
    assert isinstance(plans, list)
    for p in plans:
        assert p["quantity"] >= 1
        assert p["direction"] in ("buy", "sell")
