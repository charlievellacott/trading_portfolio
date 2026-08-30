"""Pathwise holes, joint vs-SPY shape, and realized OOS overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk.monte_carlo.geometry import (
    ev_concentration,
    excess_wealth_paths,
    joint_shape_vs_spy,
    path_max_drawdown,
    pathwise_holes,
    realized_terminal_percentile,
    realized_wealth_for_fan,
    wealth_with_start,
)
from risk.monte_carlo.report import run_ev_vs_spy


def test_max_dd_includes_start_wealth():
    paths = pd.DataFrame({"sim_0": [-0.10, -0.10]})
    w = wealth_with_start(paths)
    assert list(w["sim_0"]) == pytest.approx([1.0, 0.9, 0.81])
    dd = path_max_drawdown(paths)
    assert abs(float(dd.iloc[0]) - (0.81 - 1.0)) < 1e-12


def test_max_dd_peak_to_trough_after_gain():
    paths = pd.DataFrame({"a": [0.20, -0.25]})
    dd = float(path_max_drawdown(paths).iloc[0])
    assert abs(dd - (0.9 / 1.2 - 1.0)) < 1e-12


def test_holes_scatter_length_matches_n_sim():
    rng = np.random.default_rng(1)
    paths = pd.DataFrame(rng.normal(0.001, 0.02, size=(15, 11)))
    holes = pathwise_holes(paths)
    assert len(holes) == 11
    assert set(holes.columns) >= {
        "terminal_wealth",
        "max_dd",
        "frac_in_drawdown",
        "bars_to_recover",
    }
    assert (holes["max_dd"] <= 0.0 + 1e-12).all()


def test_joint_down_capture_uses_paired_columns():
    rng = np.random.default_rng(8)
    spy = pd.DataFrame(rng.normal(0.0, 0.02, size=(30, 6)), columns=[f"sim_{i}" for i in range(6)])
    strat = spy * 2.0
    shape = joint_shape_vs_spy(strat, spy)
    assert abs(float(shape["beta_median"]) - 2.0) < 1e-9
    assert abs(float(shape["corr_median"]) - 1.0) < 1e-9

    spy_perm = spy.copy()
    spy_perm.iloc[:, :] = spy.to_numpy()[:, ::-1]
    broken = joint_shape_vs_spy(strat, spy_perm)
    assert abs(float(broken["beta_median"]) - 2.0) > 0.5


def test_p_not_beat_given_spy_underwater():
    strat = pd.DataFrame(np.full((4, 6), 0.0))
    spy = pd.DataFrame(np.full((4, 6), -0.02))
    shape = joint_shape_vs_spy(strat, spy)
    # SPY wealth < 1 on every path; strategy flat at 1 > spy → not-beat is False
    assert shape["p_spy_terminal_underwater"] == 1.0
    assert shape["p_not_beat_given_spy_underwater"] == 0.0


def test_excess_wealth_and_realized_percentile():
    s = pd.DataFrame(np.full((5, 8), 0.01))
    b = pd.DataFrame(np.full((5, 8), 0.0))
    xs = excess_wealth_paths(s, b)
    assert xs.shape == (5, 8)
    assert (xs.iloc[-1] > 1.0).all()
    realized = realized_wealth_for_fan(pd.Series(np.full(5, 0.01)), horizon=5)
    pct = realized_terminal_percentile(realized, s)
    assert 0.0 <= pct <= 100.0


def test_ev_concentration_top_share():
    paths = pd.DataFrame(
        {
            0: np.full(4, 0.10),
            1: np.full(4, -0.02),
            2: np.full(4, -0.02),
            3: np.full(4, -0.02),
            4: np.full(4, -0.02),
            5: np.full(4, -0.02),
            6: np.full(4, -0.02),
            7: np.full(4, -0.02),
            8: np.full(4, -0.02),
            9: np.full(4, -0.02),
        }
    )
    conc = ev_concentration(paths, top_frac=0.10)
    assert conc["top_n"] == 1.0
    assert conc["mean_terminal"] < conc["top_decile_ev_share"] or conc["top_decile_ev_share"] > 1.0


def test_run_ev_vs_spy_geometry_bundle():
    rng = np.random.default_rng(3)
    n = 40
    idx = pd.bdate_range("2019-01-07", periods=n, freq="W-MON")
    spy = rng.normal(0.002, 0.018, n)
    strat = 0.001 + 0.35 * spy + rng.normal(0, 0.014, n)
    frame = pd.DataFrame({"strategy": strat, "spy": spy}, index=idx)
    pack = run_ev_vs_spy(
        frame,
        n_simulations=24,
        horizon=12,
        leverage=1.0,
        mean_block_length=4.0,
        periods_per_year=52.0,
        random_seed=3,
        n_bootstrap=80,
    )
    assert len(pack["holes"]) == 24
    assert "max_dd_median" in pack["headline"].index
    assert "p_not_beat_given_spy_underwater" in pack["joint_shape"].index
    assert pack["fan"] is not None
    assert pack["excess_fan"] is not None
    assert pack["max_dd_hist"] is not None
    assert pack["dd_scatter"] is not None
    # overlay traces exist
    names = [t.name for t in pack["fan"].data]
    assert any(n_ and "OOS strategy" in n_ for n_ in names)
    xs_names = [t.name for t in pack["excess_fan"].data]
    assert any(n_ and "OOS excess" in n_ for n_ in xs_names)
    assert 0.0 <= float(pack["headline"]["oos_terminal_percentile"]) <= 100.0
