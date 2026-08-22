"""Tests for S2 star-stack lookback helpers."""

from __future__ import annotations

from backtest.s2_coint.research import (
    DAY_OLS_WINDOW,
    DAY_Z_WINDOW,
    config_from_stack,
    lookbacks_for_bar,
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
