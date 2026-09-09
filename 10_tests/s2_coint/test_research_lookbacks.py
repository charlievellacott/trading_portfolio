"""Tests for S2 star-stack lookback helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.s2_coint.research import (
    DAY_OLS_WINDOW,
    DAY_Z_WINDOW,
    align_panels_to_common_sessions,
    clip_panel_calendar,
    config_from_stack,
    lookbacks_for_bar,
    overlap_calendar_bounds,
    overlap_is_end,
    split_is_oos,
)
from strategies.s2_coint.config import S2SimConfig


def test_lookbacks_for_bar_defaults_1d():
    lb = lookbacks_for_bar("1d")
    assert lb["ols_window"] == DAY_OLS_WINDOW
    assert lb["z_window"] == DAY_Z_WINDOW
    assert lb["adf_window"] == DAY_OLS_WINDOW


def test_lookbacks_for_bar_day_overrides_and_1h_scale():
    lb = lookbacks_for_bar("1h", ols_days=63, z_days=40, adf_days=126)
    assert lb["ols_window"] == 63 * 6
    assert lb["z_window"] == 40 * 6
    assert lb["adf_window"] == 126 * 6


def test_config_from_stack_honors_window_stars():
    stack = {
        "BAR_STAR": "1d",
        "OLS_WINDOW_STAR": 126,
        "Z_WINDOW_STAR": 40,
        "ADF_WINDOW_STAR": 63,
        "ENTRY_Z_STAR": 1.5,
        "BREAK_STAR": "flat_05",
    }
    cfg = config_from_stack(stack)
    assert isinstance(cfg, S2SimConfig)
    assert cfg.ols_window == 126
    assert cfg.z_window == 40
    assert cfg.adf_window == 63
    assert cfg.entry_z == 1.5
    assert cfg.break_mode == "flat_05"


def test_config_from_stack_soft_defaults_when_stars_unset():
    cfg = config_from_stack({"BAR_STAR": "1d", "PAIRS_STAR": []})
    assert cfg.ols_window == DAY_OLS_WINDOW
    assert cfg.z_window == DAY_Z_WINDOW
    assert cfg.adf_window == DAY_OLS_WINDOW
    assert cfg.entry_z == 2.0
    assert cfg.break_mode == "off"


def test_config_from_stack_vt_target_ann_vol_star():
    cfg = config_from_stack({"BAR_STAR": "1d", "VT_TARGET_ANN_VOL_STAR": 0.06})
    assert cfg.vt_target_ann_vol == pytest.approx(0.06)
    cfg_default = config_from_stack({"BAR_STAR": "1d", "PAIRS_STAR": []})
    assert cfg_default.vt_target_ann_vol == pytest.approx(0.10)


def test_overlap_calendar_bounds_normalize_intraday():
    # 1h starts mid-session; daily bar that day must still be in the window.
    daily = pd.Series(pd.to_datetime(["2024-09-09", "2024-09-10", "2024-09-11"]))
    hourly = pd.Series(
        pd.to_datetime(
            [
                "2024-09-09 13:30",
                "2024-09-10 10:30",
                "2024-09-11 15:30",
            ]
        )
    )
    start, end = overlap_calendar_bounds(daily, hourly)
    assert start == pd.Timestamp("2024-09-09")
    assert end == pd.Timestamp("2024-09-11")
    clipped = clip_panel_calendar(
        pd.DataFrame({"date": daily, "pair_id": "A|B"}), start, end
    )
    assert len(clipped) == 3


def test_align_panels_to_common_sessions_intersection():
    panel_1d = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-09-09", "2024-09-10", "2024-09-12"]),
            "pair_id": ["A|B"] * 3,
            "z": [0.0, 1.0, 2.0],
        }
    )
    panel_1h = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-09-09 13:30",
                    "2024-09-09 14:30",
                    "2024-09-10 10:30",
                    "2024-09-11 11:30",  # no 1d row → dropped
                ]
            ),
            "pair_id": ["A|B"] * 4,
            "z": [0.1, 0.2, 0.3, 0.4],
        }
    )
    a, b, common = align_panels_to_common_sessions(panel_1d, panel_1h)
    assert list(common.normalize()) == [
        pd.Timestamp("2024-09-09"),
        pd.Timestamp("2024-09-10"),
    ]
    assert len(a) == 2
    assert len(b) == 3
    assert pd.to_datetime(a["date"]).dt.normalize().nunique() == 2
    assert pd.to_datetime(b["date"]).dt.normalize().nunique() == 2


def test_split_is_oos_includes_full_is_end_session():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-09-09",
                    "2024-09-10 10:30",
                    "2024-09-10 15:30",
                    "2024-09-11 10:30",
                ],
                format="mixed",
            ),
            "x": [1, 2, 3, 4],
        }
    )
    is_p, oos_p = split_is_oos(panel, is_end=pd.Timestamp("2024-09-10"))
    assert len(is_p) == 3
    assert len(oos_p) == 1
    cut = overlap_is_end(panel, frac=0.70)
    assert cut == pd.Timestamp("2024-09-10")
