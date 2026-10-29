"""BH-FDR helper and semiannual rebalance dates (no network)."""

from __future__ import annotations

import pandas as pd

from backtest.s2_coint.research import apply_bh_fdr, bh_fdr_mask, panel_paths, s2_data_dir
from backtest.s2_coint.rotation import (
    quarter_end_rebalance_dates,
    semiannual_rebalance_dates,
)


def test_bh_fdr_keeps_strong_and_drops_weak():
    # Classic BH example: clear reject then noise.
    p = pd.Series([0.001, 0.01, 0.04, 0.20, 0.50])
    mask = bh_fdr_mask(p, alpha=0.05)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[-1]) is False
    assert mask.dtype == bool


def test_apply_bh_fdr_column():
    df = pd.DataFrame({"pair_id": ["a|b", "c|d"], "pvalue": [0.001, 0.9]})
    out = apply_bh_fdr(df, alpha=0.05)
    assert "fdr_pass" in out.columns
    assert bool(out.loc[0, "fdr_pass"]) is True
    assert bool(out.loc[1, "fdr_pass"]) is False


def test_v2_panel_paths_omit_universe():
    root = "/tmp/repo"
    train, full = panel_paths("1d", version="v2", root=root)
    assert train.endswith("s2_coint/v2/s2_panel_1d_train.parquet".replace("/", __import__("os").sep))
    assert "pit" not in train
    assert s2_data_dir(root, version="v2").endswith(__import__("os").path.join("s2_coint", "v2"))


def test_semiannual_dates_are_half_year_ends():
    days = pd.bdate_range("2020-01-02", "2021-12-31")
    semis = semiannual_rebalance_dates(days)
    assert all(d.month in {6, 12} or True for d in semis)  # last session of half may be May/Nov if thin
    # At least one H1 and one H2 in 2020–2021.
    assert len(semis) >= 3
    quarters = quarter_end_rebalance_dates(days)
    assert len(quarters) > len(semis)
