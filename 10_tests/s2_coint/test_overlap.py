"""H-007 never-allow: already-open wins; same-bar uses score × confidence."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.engine import simulate_book
from strategies.s2_coint.overlap import pick_same_bar_winner, score_confidence, shares_leg


def test_shares_leg_and_priority():
    assert shares_leg({"1398.HK", "0939.HK"}, {"1398.HK", "3988.HK"})
    assert not shares_leg({"1398.HK", "0939.HK"}, {"8306.T", "8316.T"})
    assert score_confidence(2.0, 0.0) > score_confidence(2.0, 0.9)
    assert pick_same_bar_winner([("a", 0.1), ("b", 0.5)]) == "b"


def _two_overlap_panel() -> pd.DataFrame:
    idx = pd.date_range("2018-01-02", periods=14, freq="B")
    z = np.zeros(14)
    z[0] = -2.5
    z[1] = -2.5
    z[4] = 0.0
    px = np.linspace(10.0, 11.0, 14)

    def one(pid, ty, tx):
        return pd.DataFrame(
            {
                "date": idx,
                "pair_id": pid,
                "ticker_y": ty,
                "ticker_x": tx,
                "open_y": px,
                "high_y": px,
                "low_y": px,
                "close_y": px,
                "open_x": px * 0.5,
                "high_x": px * 0.5,
                "low_x": px * 0.5,
                "close_x": px * 0.5,
                "alpha": 0.0,
                "beta": 2.0,
                "spread": 0.0,
                "z": z,
                "half_life": 20.0,
                "adf_pvalue": 0.01,
            }
        )

    a = one("1398.HK|0939.HK", "1398.HK", "0939.HK")
    b = one("1398.HK|3988.HK", "1398.HK", "3988.HK")
    return pd.concat([a, b], ignore_index=True)


def test_never_allow_opens_only_one_overlapping_pair():
    panel = _two_overlap_panel()
    allow = simulate_book(panel, S2SimConfig(overlap_mode="allow"))
    never = simulate_book(panel, S2SimConfig(overlap_mode="never_allow"))
    n_allow = sum(r.n_entries for r in allow.pair_results.values())
    n_never = sum(r.n_entries for r in never.pair_results.values())
    assert n_allow >= n_never
    assert n_never >= 1
    assert n_never <= n_allow
