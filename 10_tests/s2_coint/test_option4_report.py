"""Option 4 report helpers: arm_selection_table and inference columns."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.report import arm_selection_table, median_sharpe_hint
from backtest.s2_coint.research import (
    cumulative_trials_before,
    hypothesis_tier,
    load_variant_ledger,
    n_trials_stack,
)


def test_hypothesis_tier_map():
    assert hypothesis_tier("H-009") == "A"
    assert hypothesis_tier("H-006") == "B"
    assert hypothesis_tier("H-001") == "C"


def test_arm_selection_table_no_winner_column():
    fold_df = pd.DataFrame(
        {
            "arm": ["a", "a", "b", "b"],
            "fold_id": [0, 1, 0, 1],
            "ann_sharpe": [1.0, 2.0, 0.5, 1.5],
            "max_drawdown": [-0.1, -0.2, -0.15, -0.25],
            "corr_to_s1": [0.1, 0.2, -0.1, -0.2],
        }
    )
    full_is = pd.DataFrame({"arm": ["a", "b"], "full_is_sharpe": [1.5, 1.0]})
    tbl = arm_selection_table(fold_df, full_is)
    assert "winner" not in tbl.columns
    assert set(tbl["arm"]) == {"a", "b"}
    assert tbl.loc[tbl["arm"] == "a", "median_val_sharpe"].iloc[0] == pytest.approx(1.5)
    assert tbl.loc[tbl["arm"] == "a", "full_is_sharpe"].iloc[0] == pytest.approx(1.5)
    assert "max_drawdown" in tbl.columns
    assert "corr_to_s1" in tbl.columns


def test_arm_selection_table_includes_inference_columns():
    fold_df = pd.DataFrame(
        {
            "arm": ["a", "a", "b", "b"],
            "fold_id": [0, 1, 0, 1],
            "ann_sharpe": [1.0, 2.0, 0.5, 1.5],
            "max_drawdown": [-0.1, -0.2, -0.15, -0.25],
            "corr_to_s1": [0.1, 0.2, -0.1, -0.2],
            "psr": [0.6, 0.7, 0.4, 0.5],
            "dsr_local": [0.5, 0.6, 0.3, 0.4],
            "dsr_stack": [0.4, 0.5, 0.2, 0.3],
        }
    )
    full_is = pd.DataFrame(
        {
            "arm": ["a", "b"],
            "full_is_sharpe": [1.5, 1.0],
            "full_is_psr": [0.65, 0.45],
            "full_is_dsr_local": [0.55, 0.35],
            "full_is_dsr_stack": [0.45, 0.25],
        }
    )
    tbl = arm_selection_table(fold_df, full_is)
    for col in (
        "max_drawdown",
        "corr_to_s1",
        "median_psr",
        "median_dsr_local",
        "median_dsr_stack",
        "full_is_psr",
        "full_is_dsr_local",
        "full_is_dsr_stack",
    ):
        assert col in tbl.columns


def test_median_sharpe_hint_is_commentary_only():
    fold_df = pd.DataFrame(
        {"arm": ["x", "x", "y"], "fold_id": [0, 1, 0], "ann_sharpe": [1.0, 1.0, 2.0]}
    )
    assert median_sharpe_hint(fold_df) == "y"


def test_variant_ledger_seeded():
    ledger = load_variant_ledger()
    assert ledger["cumulative_arms"] == 46
    assert cumulative_trials_before("H-009") == 26
    assert n_trials_stack("H-009", {"allow": 1, "never_allow": 2}) == 28
