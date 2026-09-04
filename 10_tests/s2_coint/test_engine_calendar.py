"""Book return calendar densification (never_allow vs allow same n_days)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.engine import simulate_book
from strategies.s2_coint.metrics import book_returns_to_calendar, panel_session_dates


def _toy_panel(n: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=n)
    rows = []
    for pid, (ty, tx) in (("A|B", ("AAA", "BBB")), ("C|D", ("CCC", "DDD"))):
        beta = 1.0
        for i, dt in enumerate(dates):
            z = float(np.sin(i / 5.0) * 2.5)
            rows.append(
                {
                    "date": dt,
                    "pair_id": pid,
                    "ticker_y": ty,
                    "ticker_x": tx,
                    "open_y": 100.0 + i * 0.1,
                    "open_x": 50.0 + i * 0.05,
                    "close_y": 100.0 + i * 0.1,
                    "close_x": 50.0 + i * 0.05,
                    "high_y": 101.0,
                    "low_y": 99.0,
                    "high_x": 51.0,
                    "low_x": 49.0,
                    "beta": beta,
                    "z": z,
                    "adf_pvalue": 0.01,
                    "half_life": 10.0,
                    "spread": z,
                }
            )
    return pd.DataFrame(rows)


def test_book_returns_to_calendar_fills_zeros():
    idx = pd.bdate_range("2020-01-02", periods=5)
    sparse = pd.Series([0.01, 0.02], index=idx[[0, 3]], dtype=float, name="ret")
    dense = book_returns_to_calendar(sparse, idx)
    assert len(dense) == 5
    assert dense.iloc[1] == 0.0
    assert dense.iloc[0] == pytest.approx(0.01)


def test_never_allow_same_index_length_as_allow():
    panel = _toy_panel(40)
    cfg_allow = S2SimConfig(overlap_mode="allow", entry_z=1.5, k_in=1.5)
    cfg_never = S2SimConfig(overlap_mode="never_allow", entry_z=1.5, k_in=1.5)
    cal = panel_session_dates(panel)
    res_allow = simulate_book(panel, cfg_allow)
    res_never = simulate_book(panel, cfg_never)
    assert len(res_allow.returns) == len(cal)
    assert len(res_never.returns) == len(cal)
    assert len(res_allow.returns) == len(res_never.returns)
    assert res_never.returns.index.equals(cal)
    assert len(res_allow.returns_base) == len(res_allow.returns)
    assert res_allow.returns_base.index.equals(res_allow.returns.index)


def test_s1_vt_persists_returns_base():
    panel = _toy_panel(40)
    cfg = S2SimConfig(
        overlap_mode="allow",
        entry_z=1.5,
        k_in=1.5,
        vol_mode="s1_vt",
        vt_target_ann_vol=0.10,
    )
    res = simulate_book(panel, cfg)
    assert len(res.returns_base) == len(res.returns)
    assert res.returns_base.index.equals(res.returns.index)
    assert res.returns_base.name == "ret" or res.returns_base.name is not None


def test_simulate_book_from_stack_cfg():
    from backtest.s2_coint.research import DEFAULT_STAR_STACK, config_from_stack
    from backtest.star_stack_io import load_star_stack

    stack = load_star_stack(DEFAULT_STAR_STACK)
    cfg = config_from_stack(
        stack, z_window=10, ols_window=20, adf_window=20, sigma_window=20
    )
    assert cfg.vt_target_ann_vol == pytest.approx(0.06)
    res = simulate_book(_toy_panel(40), cfg)
    assert len(res.returns) == len(panel_session_dates(_toy_panel(40)))
    assert len(res.returns_base) == len(res.returns)


def test_panel_session_dates_sorted_unique():
    panel = _toy_panel(10)
    cal = panel_session_dates(panel)
    assert cal.is_monotonic_increasing
    assert len(cal) == 10
