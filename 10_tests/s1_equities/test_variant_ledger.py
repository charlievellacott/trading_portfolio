"""S1 variant ledger registration."""

from __future__ import annotations

import os

from backtest.s1_equities.research import (
    load_variant_ledger,
    n_trials_ledger_total,
    n_trials_stack,
    register_hypothesis_arms,
)


def test_register_and_stack_count(tmp_path):
    path = os.path.join(str(tmp_path), "s1_variant_ledger.json")
    register_hypothesis_arms("H-001", ["a", "b"], path=path)
    register_hypothesis_arms("H-002", ["c"], path=path)
    ledger = load_variant_ledger(path)
    assert ledger["cumulative_arms"] == 3
    assert n_trials_ledger_total(path=path) == 3
    assert n_trials_stack("H-002", {"c": 1}, path=path) == 3


def test_shipped_ledger_has_arms():
    total = n_trials_ledger_total()
    assert total >= 1
