"""H-010 orthogonal exits: ATR multiple sizing + stack mapping."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.research import config_from_stack
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.engine import simulate_book, simulate_pair
from strategies.s2_coint.sizing import atr_size_multiplier


def test_atr_size_multiplier_scales_with_mult():
    base = atr_size_multiplier(
        atr=2.0, beta=1.0, n_pairs=1, pair_scale=1.0, leverage=1.0, risk_frac=0.01, atr_mult=1.0
    )
    wide = atr_size_multiplier(
        atr=2.0, beta=1.0, n_pairs=1, pair_scale=1.0, leverage=1.0, risk_frac=0.01, atr_mult=2.0
    )
    assert wide == pytest.approx(base / 2.0)


def test_config_orthogonal_exit_defaults():
    cfg = config_from_stack({"BAR_STAR": "1d"})
    assert cfg.atr_stop_mult is None
    assert cfg.hl_exit_n is None
    assert cfg.pair_max_loss == pytest.approx(-0.10)
    assert not cfg.atr_stop_enabled()
    assert not cfg.hl_exit_enabled()


def test_config_from_stack_atr_and_hl_stars():
    cfg = config_from_stack(
        {
            "BAR_STAR": "1d",
            "ATR_EXIT_STAR": 2.5,
            "HL_EXIT_STAR": "on",
            "PAIR_MAX_LOSS_STAR": -0.10,
        }
    )
    assert cfg.atr_stop_mult == pytest.approx(2.5)
    assert cfg.hl_exit_n == pytest.approx(3.0)
    assert cfg.atr_stop_enabled()
    assert cfg.hl_exit_enabled()


def test_config_from_stack_legacy_exit_star():
    cfg = config_from_stack({"BAR_STAR": "1d", "EXIT_STAR": "hl3_atr_breaker"})
    assert cfg.atr_stop_mult == pytest.approx(1.0)
    assert cfg.hl_exit_n == pytest.approx(3.0)


def test_s2simconfig_rejects_nonpositive_atr_mult():
    with pytest.raises(ValueError):
        S2SimConfig(atr_stop_mult=0.0)


def _atr_stop_panel(
    *,
    side: str,
    n: int = 25,
    close_breach: bool,
) -> pd.DataFrame:
    """Synthetic pair with ATR cols; optional close breach after entry.

    Long entry when z <= -2; short when z >= +2. Stop at entry_open ∓ 1×ATR.
    Extreme H/L always pierce the stop; only ``spread_close`` decides under close-t rules.
    """
    dates = pd.bdate_range("2020-01-02", periods=n)
    z_entry = -2.5 if side == "long" else 2.5
    z_hold = -1.0 if side == "long" else 1.0
    # Entry at signal i=5 → fill open i=6. First close check at i=6.
    rows = []
    for i, dt in enumerate(dates):
        z = z_entry if i <= 5 else z_hold
        # Entry open / ATR fixed so stop is known: long stop=-1, short stop=+1.
        so = 0.0
        atr = 1.0
        if side == "long":
            # H/L would always stop a long; close only if close_breach.
            sc = -1.5 if (close_breach and i >= 6) else -0.25
            sl, sh = -5.0, 5.0
        else:
            sc = 1.5 if (close_breach and i >= 6) else 0.25
            sl, sh = -5.0, 5.0
        px = 100.0 + i * 0.01
        rows.append(
            {
                "date": dt,
                "pair_id": "AAA|BBB",
                "ticker_y": "AAA",
                "ticker_x": "BBB",
                "open_y": px,
                "high_y": px + 1.0,
                "low_y": px - 1.0,
                "close_y": px,
                "open_x": px / 2.0,
                "high_x": px / 2.0 + 0.5,
                "low_x": px / 2.0 - 0.5,
                "close_x": px / 2.0,
                "alpha": 0.0,
                "beta": 1.0,
                "spread": float(sc),
                "spread_open": so,
                "spread_high": sh,
                "spread_low": sl,
                "spread_close": float(sc),
                "atr_spread": atr,
                "z": z,
                "half_life": 20.0,
                "adf_pvalue": 0.01,
                "variance_jump": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _atr_cfg(**overrides) -> S2SimConfig:
    kwargs = dict(
        atr_stop_mult=1.0,
        pair_max_loss=-1e9,  # isolate ATR stop from breaker
        break_mode="off",
        entry_z=2.0,
        exit_z=0.0,
        size_mode="equal",
        vol_mode="fixed_k",
        overlap_mode="allow",
        trend_mode="off",
        hl_exit_n=None,
    )
    kwargs.update(overrides)
    return S2SimConfig(**kwargs)


def test_atr_stop_long_triggers_on_spread_close_not_hl():
    cfg = _atr_cfg()
    # Extreme low always present; close does not breach → no atr_stop.
    no_hit = simulate_pair(_atr_stop_panel(side="long", close_breach=False), cfg)
    reasons = set(no_hit.trades["exit_reason"].tolist()) if not no_hit.trades.empty else set()
    assert "atr_stop" not in reasons

    hit = simulate_pair(_atr_stop_panel(side="long", close_breach=True), cfg)
    assert not hit.trades.empty
    assert "atr_stop" in set(hit.trades["exit_reason"].tolist())


def test_atr_stop_short_triggers_on_spread_close_not_hl():
    cfg = _atr_cfg()
    no_hit = simulate_pair(_atr_stop_panel(side="short", close_breach=False), cfg)
    reasons = set(no_hit.trades["exit_reason"].tolist()) if not no_hit.trades.empty else set()
    assert "atr_stop" not in reasons

    hit = simulate_pair(_atr_stop_panel(side="short", close_breach=True), cfg)
    assert not hit.trades.empty
    assert "atr_stop" in set(hit.trades["exit_reason"].tolist())


def test_atr_stop_close_rule_in_never_allow_book():
    """Joint book path (never_allow) must use the same close-t ATR rule."""
    cfg = _atr_cfg(overlap_mode="never_allow")
    panel = _atr_stop_panel(side="long", close_breach=True)
    book = simulate_book(panel, cfg)
    trades = book.pair_results["AAA|BBB"].trades
    assert not trades.empty
    assert "atr_stop" in set(trades["exit_reason"].tolist())
