"""S1Strategy reads frozen STARs from the sleeve star stack."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s1_equities.research import DEFAULT_STAR_STACK
from backtest.star_stack_io import load_star_stack, save_star_stack
from strategies.s1_equities.s1_strategy import S1Strategy


def test_default_s1_stack_has_desk_values():
    stack = load_star_stack(DEFAULT_STAR_STACK)
    assert int(stack["N_STAR"]) == 15
    assert stack["VT_STAR"] == "vt_bayes_0.9_10_q0.75_db0.05"
    assert float(stack["VT_TARGET_ANN_VOL_STAR"]) == pytest.approx(0.10)


def test_s1_strategy_reads_custom_stack(tmp_path):
    stack = load_star_stack(DEFAULT_STAR_STACK)
    stack["N_STAR"] = 11
    stack["INV_VOL_WINDOW_STAR"] = 21
    stack["VT_TARGET_ANN_VOL_STAR"] = 0.08
    path = os.path.join(str(tmp_path), "s1_star_stack.json")
    save_star_stack(path, stack)
    strategy = S1Strategy("2024-01-08", star_stack_path=path)
    assert strategy.N_STAR == 11
    assert strategy.INV_VOL_WINDOW_STAR == 21
    assert strategy.vt_target_ann_vol == pytest.approx(0.08)
    assert strategy.VT_STAR == stack["VT_STAR"]
    assert len(strategy.tickers) == 97
