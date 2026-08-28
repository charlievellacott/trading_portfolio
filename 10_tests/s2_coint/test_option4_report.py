"""Option 4 report helpers: arm_selection_table and full_is_metrics."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.report import arm_selection_table, median_sharpe_hint
from backtest.s2_coint.research import hypothesis_tier


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


def test_median_sharpe_hint_is_commentary_only():
    fold_df = pd.DataFrame(
        {"arm": ["x", "x", "y"], "fold_id": [0, 1, 0], "ann_sharpe": [1.0, 1.0, 2.0]}
    )
    assert median_sharpe_hint(fold_df) == "y"
