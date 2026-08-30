"""FTMO 2-step rule tests (examples from the official 2-step wording)."""

from __future__ import annotations

import pandas as pd

from risk.prop_firm.ftmo_two_step import (
    FtmoTwoStepChallenge,
    FtmoTwoStepFunded,
    FtmoTwoStepVerification,
    run_two_step,
)
from risk.prop_firm.registry import CHALLENGES, make_challenge


def _path(*rets: float) -> pd.Series:
    return pd.Series(list(rets), dtype=float)


def test_registry_keys():
    assert "ftmo.2step.challenge" in CHALLENGES
    assert "ftmo.2step.verification" in CHALLENGES
    assert "ftmo.2step.funded" in CHALLENGES
    chal = make_challenge("ftmo.2step.challenge")
    assert chal.profit_target_frac() == 0.10
    assert make_challenge("ftmo.2step.verification").profit_target_frac() == 0.05
    assert make_challenge("ftmo.2step.funded").profit_target_frac() is None


def test_day1_daily_limit_95000():
    chal = FtmoTwoStepChallenge()
    cap = 100_000.0
    assert chal.daily_limit(cap, cap) == 95_000.0
    assert chal.max_loss_floor(cap) == 90_000.0
    # equity 96k: ok; 94k: daily fail
    ok = chal.evaluate_path(_path(-0.04), initial_capital=cap)
    assert ok.status == "incomplete" or ok.status in ("open", "incomplete")
    # -4% -> 96k, above 95k, target not hit
    assert ok.status == "incomplete"
    fail = chal.evaluate_path(_path(-0.06), initial_capital=cap)
    assert fail.status == "failed"
    assert fail.first_binding == "daily_loss"
    assert fail.failure_reason == "daily_loss"


def test_day2_daily_limit_from_midnight_balance():
    chal = FtmoTwoStepChallenge()
    cap = 100_000.0
    # Day 1 +2% -> 102_000; midnight SOD = 102_000; limit = 97_000
    # 102000 * (1+r) = 96900 -> r = 96900/102000 - 1
    r2 = 96_900.0 / 102_000.0 - 1.0
    out = chal.evaluate_path(_path(0.02, r2), initial_capital=cap)
    assert out.status == "failed"
    assert out.first_binding == "daily_loss"


def test_static_max_loss_floor_90000():
    chal = FtmoTwoStepChallenge()
    cap = 100_000.0
    assert chal.max_loss_floor(cap) == 90_000.0
    # Gap through both floors: first binding is the higher floor (daily 95k)
    gap = chal.evaluate_path(_path(-0.11), initial_capital=cap)
    assert gap.status == "failed"
    assert gap.min_equity < 90_000.0
    assert gap.first_binding == "daily_loss"


def test_max_loss_binds_when_daily_limit_is_looser():
    """Widen daily amount so the static 10% floor is hit first."""
    phase = FtmoTwoStepChallenge(max_daily_loss_frac=0.20)
    cap = 100_000.0
    # Day-1 daily limit = 80k, max floor = 90k; close at 85k
    out = phase.evaluate_path(_path(-0.15), initial_capital=cap)
    assert out.status == "failed"
    assert out.first_binding == "max_loss"


def test_profit_target_closed_balance_and_min_four_open_days():
    chal = FtmoTwoStepChallenge()
    cap = 100_000.0
    # +15% day 1 hits 10% target but only 1 trading day
    early = chal.evaluate_path(_path(0.15, 0.0, 0.0, 0.0), initial_capital=cap)
    assert early.passed is False
    assert early.status == "incomplete"
    assert early.target_hit_bar == 0
    assert early.trading_days_opened == 1
    # four open days at ~3% compounds past 10%
    ok = chal.evaluate_path(_path(0.03, 0.03, 0.03, 0.03), initial_capital=cap)
    assert ok.passed is True
    assert ok.status == "passed"
    assert ok.days_to_pass == 4
    assert ok.trading_days_opened >= 4
    assert ok.final_balance >= 110_000.0


def test_verification_five_percent_vs_challenge_ten():
    cap = 100_000.0
    # ~5.3% over 4 days
    rets = _path(0.013, 0.013, 0.013, 0.013)
    ver = FtmoTwoStepVerification().evaluate_path(rets, initial_capital=cap)
    chal = FtmoTwoStepChallenge().evaluate_path(rets, initial_capital=cap)
    assert ver.passed is True
    assert chal.passed is False
    assert chal.status == "incomplete"


def test_funded_no_target_no_min_days():
    cap = 100_000.0
    fund = FtmoTwoStepFunded()
    out = fund.evaluate_path(_path(0.01, 0.01, 0.01, 0.01), initial_capital=cap)
    assert out.passed is False
    assert out.status == "survived"
    assert fund.profit_target_frac() is None
    assert fund.min_trading_days() == 0
    blow = fund.evaluate_path(_path(-0.06), initial_capital=cap)
    assert blow.status == "failed"


def test_incomplete_is_not_fail():
    chal = FtmoTwoStepChallenge()
    out = chal.evaluate_path(_path(0.01, 0.01, 0.01, 0.01), initial_capital=100_000.0)
    assert out.status == "incomplete"
    assert out.passed is False
    assert out.failure_reason is None


def test_two_step_resets_capital():
    cap = 100_000.0
    # Challenge path: four +3% days -> pass
    chal_p = pd.DataFrame({"sim_0": [0.03, 0.03, 0.03, 0.03]})
    # Verification: fail daily on day 1
    ver_p = pd.DataFrame({"sim_0": [-0.06, 0.0, 0.0, 0.0]})
    fund_p = pd.DataFrame({"sim_0": [0.0, 0.0, 0.0, 0.0]})
    out = run_two_step(chal_p, ver_p, fund_p, initial_capital=cap)
    assert bool(out.iloc[0]["passed_challenge"]) is True
    assert bool(out.iloc[0]["passed_verification"]) is False
    assert bool(out.iloc[0]["passed_both"]) is False
    assert out.iloc[0]["funded_surplus"] == 0.0


def test_two_step_both_pass_uses_funded_surplus():
    cap = 100_000.0
    chal_p = pd.DataFrame({"a": [0.03, 0.03, 0.03, 0.03]})
    ver_p = pd.DataFrame({"a": [0.02, 0.02, 0.01, 0.01]})
    fund_p = pd.DataFrame({"a": [0.05, 0.0, 0.0, 0.0]})
    out = run_two_step(chal_p, ver_p, fund_p, initial_capital=cap)
    row = out.iloc[0]
    assert bool(row["passed_both"]) is True
    assert row["funded_status"] == "survived"
    assert abs(row["funded_surplus"] - 5_000.0) < 1e-6


def test_pay_once_ev_do_not_take():
    from risk.prop_firm.economics import attach_economics

    df = pd.DataFrame(
        {
            "passed_both": [True, False, False, True],
            "funded_surplus": [10_000.0, 0.0, 0.0, 2_000.0],
        }
    )
    _cash, summary = attach_economics(
        df, fee=1_000.0, profit_split=0.8, horizon_days=40.0
    )
    # two payouts: 8000 and 1600; four fees of 1000 → EV = (8000+1600)/4 - 1000 = 1400
    assert abs(float(summary["ev"]) - 1400.0) < 1e-6
    assert "p_ev_le_0" in summary.index
    assert "do_not_take" in summary.index


def test_leverage_grid_fail_rate_columns_and_ftmo_cap():
    from risk.prop_firm.report import suggest_challenge_leverage

    grid = pd.DataFrame(
        {
            "leverage": [0.5, 1.0, 2.0],
            "ev_per_day": [1.0, 3.0, 10.0],
            "p_both": [0.2, 0.3, 0.1],
            "p_fail_daily_loss": [0.10, 0.20, 0.80],
            "p_fail_max_loss": [0.05, 0.10, 0.20],
            "do_not_take": [False, False, False],
        }
    )
    sug = suggest_challenge_leverage(
        grid, k_fair=1.5, max_p_fail_daily=0.30, max_p_fail_max=0.30
    )
    assert sug["k_ftmo"] == 1.0
    assert sug["k_suggested"] == 1.0
    assert sug["k_suggested"] <= sug["k_ftmo"]
    row = grid.loc[grid["leverage"] == sug["k_ftmo"]].iloc[0]
    assert float(row["p_fail_daily_loss"]) <= 0.30
    assert float(row["p_fail_max_loss"]) <= 0.30


def test_suggested_k_never_exceeds_fail_rate_survivor():
    from risk.prop_firm.report import suggest_challenge_leverage

    grid = pd.DataFrame(
        {
            "leverage": [0.25, 0.75, 1.5],
            "ev_per_day": [0.5, 2.0, 9.0],
            "p_both": [0.1, 0.2, 0.05],
            "p_fail_daily_loss": [0.05, 0.15, 0.90],
            "p_fail_max_loss": [0.02, 0.08, 0.40],
            "do_not_take": [False, False, False],
        }
    )
    sug = suggest_challenge_leverage(
        grid, k_fair=3.0, max_p_fail_daily=0.20, max_p_fail_max=0.20
    )
    assert sug["k_ftmo"] == 0.75
    assert sug["k_suggested"] == 0.75
    assert sug["k_suggested"] < 1.5


def test_leverage_ev_grid_includes_fail_rate_columns():
    from risk.prop_firm.report import leverage_ev_grid

    idx = pd.bdate_range("2020-01-01", periods=40)
    r = pd.Series(0.002, index=idx)
    grid = leverage_ev_grid(
        r,
        [0.5, 1.0],
        n_simulations=6,
        horizon=8,
        horizon_funded=5,
        initial_capital=100_000.0,
        fee=540.0,
        profit_split=0.8,
        mean_block_length=4.0,
        random_seed=3,
    )
    assert "p_fail_daily_loss" in grid.columns
    assert "p_fail_max_loss" in grid.columns
    assert list(grid["leverage"]) == [0.5, 1.0]
