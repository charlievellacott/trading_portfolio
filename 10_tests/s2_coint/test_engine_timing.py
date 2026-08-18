"""S2 engine timing: close t / fill t+1; default config matches H-001 baseline."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.report import require_star
from backtest.s2_coint.runner import run_s2_backtest
from strategies.s2_coint.baseline import simulate_pair_baseline
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.engine import simulate_book, simulate_pair


def _hk_panel(*, n: int = 12, z: np.ndarray | None = None) -> pd.DataFrame:
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    if z is None:
        z = np.zeros(n, dtype=float)
        z[0] = -2.5
        z[1] = -2.0
        z[2] = 0.0
    px = np.linspace(10.0, 11.0, n)
    return pd.DataFrame(
        {
            "date": idx,
            "pair_id": "1398.HK|0939.HK",
            "ticker_y": "1398.HK",
            "ticker_x": "0939.HK",
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
            "adf_pvalue": np.linspace(0.20, 0.01, n),
        }
    )


def test_default_engine_matches_baseline():
    panel = _hk_panel()
    base = simulate_pair_baseline(panel)
    eng = simulate_pair(panel, S2SimConfig())
    pd.testing.assert_series_equal(base.returns, eng.returns, check_names=True)
    assert eng.n_entries == base.n_entries
    assert len(eng.trades) == len(base.trades)


def test_no_same_bar_close_fill_column():
    panel = _hk_panel()
    assert "open_y_lead" not in panel.columns
    res = run_s2_backtest(panel, S2SimConfig())
    assert not res.returns.empty
    # fill dates are the day after the signal (second bar onward)
    assert res.returns.index.min() > panel["date"].min()


def test_require_star_blocks_none():
    with pytest.raises(ValueError, match="None"):
        require_star("BAR_STAR", None)
    require_star("BAR_STAR", "1d")


def test_config_rejects_4h():
    with pytest.raises(ValueError, match="4h"):
        S2SimConfig(bar="4h")
