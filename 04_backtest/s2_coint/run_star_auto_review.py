"""One-shot H-003..H-015 fold-val star review (not for production auto-pick)."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.diagnosis import gross_returns_from_net
from backtest.s2_coint.report import (
    arm_selection_table,
    fold_val_metrics,
    load_star_stack,
    median_sharpe_hint,
    save_star_stack,
)
from backtest.s2_coint.research import (
    DAY_OLS_WINDOW,
    DAY_Z_WINDOW,
    DEFAULT_STAR_STACK,
    config_from_stack,
    frozen_pairs_for_universe,
    is_end_for_stack,
    load_s1_weekly,
    load_universe_panels,
    lookbacks_for_bar,
    overlay_kalman_hedge,
    overlay_ols_hedge,
    register_hypothesis_arms,
    split_is_oos,
)
from backtest.s2_coint.rotation import (
    freeze_book,
    quarterly_rotate_schedule,
    simulate_frozen_book,
    simulate_rotating_book,
)
from backtest.s2_coint.walkforward import embargo_bars_for_config, make_s2_folds
from strategies.s2_coint.metrics import corr_to_s1, metrics_from_returns

OLS_DAY_GRID = (504, 252, 126, 63)
ADF_DAY_GRID = (504, 252, 126, 63)
ENTRY_Z_GRID = (1.5, 2.0, 2.5)
Z_DAY_GRID = (40, 60, 90)
KALMAN_DELTA_GRID = (1e-4, 1e-5, 1e-6, 1e-7)
BREAK_ARMS = ("off", "block_05_flat_10", "flat_05")


def _agg(fold_df: pd.DataFrame, sharpe_col: str = "ann_sharpe") -> pd.DataFrame:
    return fold_df.groupby("arm").agg(
        median_sharpe=(sharpe_col, "median"),
        median_dd=("max_drawdown", "median"),
        median_corr=("corr_to_s1", "median"),
    )


def pick_arm(fold_df: pd.DataFrame, *, sharpe_col: str = "ann_sharpe") -> str:
    agg = _agg(fold_df, sharpe_col)
    ranked = agg.sort_values(
        ["median_sharpe", "median_dd", "median_corr"],
        ascending=[False, False, True],
    )
    return str(ranked.index[0])


def prepare_panel(stack: dict, *, use_stack_windows: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    universe = str(stack["UNIVERSE_STAR"])
    bar = str(stack["BAR_STAR"])
    pairs = list(stack.get("PAIRS_STAR") or frozen_pairs_for_universe(universe, bar, root=ROOT))
    train, full = load_universe_panels(universe, bar, pairs, root=ROOT)
    if use_stack_windows and stack.get("OLS_WINDOW_STAR"):
        lb = lookbacks_for_bar(
            bar,
            ols_days=int(stack["OLS_WINDOW_STAR"]),
            z_days=int(stack.get("Z_WINDOW_STAR") or DAY_Z_WINDOW),
            adf_days=int(stack.get("ADF_WINDOW_STAR") or DAY_OLS_WINDOW),
        )
    else:
        lb = lookbacks_for_bar(bar)
    train = overlay_ols_hedge(
        train,
        ols_window=lb["ols_window"],
        z_window=lb["z_window"],
        hl_window=lb["hl_window"],
        adf_window=lb["adf_window"],
    )
    full = overlay_ols_hedge(
        full,
        ols_window=lb["ols_window"],
        z_window=lb["z_window"],
        hl_window=lb["hl_window"],
        adf_window=lb["adf_window"],
    )
    if stack.get("HEDGE_STAR") == "kalman" and stack.get("KALMAN_DELTA_STAR") is not None:
        delta = float(stack["KALMAN_DELTA_STAR"])
        train = overlay_kalman_hedge(
            train,
            burn_in=lb["kalman_burn_in"],
            z_window=lb["z_window"],
            hl_window=lb["hl_window"],
            adf_window=lb["adf_window"],
            delta=delta,
        )
        full = overlay_kalman_hedge(
            full,
            burn_in=lb["kalman_burn_in"],
            z_window=lb["z_window"],
            hl_window=lb["hl_window"],
            adf_window=lb["adf_window"],
            delta=delta,
        )
    is_end = is_end_for_stack(stack, full if bar == "1h" else train)
    if bar == "1d":
        is_panel = train.copy()
        oos_panel = full.loc[pd.to_datetime(full["date"]) > is_end].copy()
    else:
        is_panel, oos_panel = split_is_oos(full, is_end=is_end)
    s1 = load_s1_weekly(ROOT)
    return is_panel, oos_panel, s1


def make_folds(is_panel: pd.DataFrame, bar: str) -> list:
    dates = pd.DatetimeIndex(pd.to_datetime(is_panel["date"])).sort_values().unique()
    return make_s2_folds(dates, n_folds=3, embargo_bars=embargo_bars_for_config(bar=bar))


def run_h003(stack: dict, s1: pd.Series) -> tuple[str, pd.DataFrame]:
    universe = str(stack["UNIVERSE_STAR"])
    bar = str(stack["BAR_STAR"])
    pairs = frozen_pairs_for_universe(universe, bar, root=ROOT)
    train, full = load_universe_panels(universe, bar, pairs, root=ROOT)
    lb0 = lookbacks_for_bar(bar)
    train = overlay_ols_hedge(
        train,
        ols_window=lb0["ols_window"],
        z_window=lb0["z_window"],
        hl_window=lb0["hl_window"],
        adf_window=lb0["adf_window"],
    )
    full = overlay_ols_hedge(
        full,
        ols_window=lb0["ols_window"],
        z_window=lb0["z_window"],
        hl_window=lb0["hl_window"],
        adf_window=lb0["adf_window"],
    )
    is_end = is_end_for_stack(stack, train)
    is_panel = train.copy()
    folds = make_folds(is_panel, bar)
    configs = {arm: config_from_stack(stack, break_mode=arm) for arm in BREAK_ARMS}
    fold_df = fold_val_metrics(is_panel, folds, configs, hyp_id="H-003", s1_weekly=s1)
    register_hypothesis_arms("H-003", list(BREAK_ARMS), overwrite=True)
    return pick_arm(fold_df), fold_df


def run_h004(stack: dict, s1: pd.Series) -> tuple[str, list[str] | None, pd.DataFrame]:
    universe = str(stack["UNIVERSE_STAR"])
    bar = str(stack["BAR_STAR"])
    is_panel, _, s1 = prepare_panel(stack, use_stack_windows=False)
    lb0 = lookbacks_for_bar(bar)
    is_end = is_end_for_stack(stack, is_panel)
    cfg_base = config_from_stack(stack)
    freeze_sel = freeze_book(
        is_panel,
        is_end=is_end,
        ols_window=lb0["ols_window"],
        adf_window=lb0["adf_window"],
        root=ROOT,
    )
    schedule = quarterly_rotate_schedule(
        is_panel,
        is_end=is_end,
        ols_window=lb0["ols_window"],
        adf_window=lb0["adf_window"],
        root=ROOT,
    )
    folds = make_folds(is_panel, bar)
    rows = []
    for f in folds:
        fold_panel = is_panel.loc[pd.to_datetime(is_panel["date"]).isin(f.val_dates)]
        frozen = simulate_frozen_book(fold_panel, list(freeze_sel["pair_id"]), cfg_base)
        rotating = simulate_rotating_book(fold_panel, schedule, cfg_base)
        for arm, res in (("freeze", frozen), ("rotate", rotating)):
            net = res["returns"]
            gross = gross_returns_from_net(net)
            m_net = metrics_from_returns(net)
            m_gross = metrics_from_returns(gross)
            rows.append(
                {
                    "arm": arm,
                    "fold_id": f.fold_id,
                    "ann_sharpe": m_net["ann_sharpe"],
                    "max_drawdown": m_net["max_drawdown"],
                    "corr_to_s1": corr_to_s1(net, s1),
                    "ann_sharpe_net": m_net["ann_sharpe"],
                    "ann_sharpe_gross": m_gross["ann_sharpe"],
                }
            )
    fold_df = pd.DataFrame(rows)
    register_hypothesis_arms("H-004", ["freeze", "rotate"], overwrite=True)
    winner = pick_arm(fold_df, sharpe_col="ann_sharpe")
    pairs = list(freeze_sel["pair_id"]) if winner == "freeze" else None
    return winner, pairs, fold_df


def screen_days(
    stack: dict,
    is_panel: pd.DataFrame,
    folds: list,
    s1: pd.Series,
    *,
    grid: tuple[int, ...],
    kind: str,
    ols_days: int,
    adf_days: int,
    z_days: int,
) -> tuple[int, pd.DataFrame]:
    bar = str(stack["BAR_STAR"])
    rows = []
    for days in grid:
        if kind == "ols":
            lb = lookbacks_for_bar(bar, ols_days=days, z_days=z_days, adf_days=adf_days)
        elif kind == "adf":
            lb = lookbacks_for_bar(bar, ols_days=ols_days, z_days=z_days, adf_days=days)
        else:
            lb = lookbacks_for_bar(bar, ols_days=ols_days, z_days=days, adf_days=adf_days)
        p = overlay_ols_hedge(
            is_panel,
            ols_window=lb["ols_window"],
            z_window=lb["z_window"],
            hl_window=lb["hl_window"],
            adf_window=lb["adf_window"],
        )
        cfg = config_from_stack(
            stack,
            ols_window=lb["ols_window"],
            z_window=lb["z_window"],
            adf_window=lb["adf_window"],
        )
        part = fold_val_metrics(p, folds, {str(days): cfg}, s1_weekly=s1)
        rows.append(part)
    fold_df = pd.concat(rows, ignore_index=True)
    winner = int(pick_arm(fold_df))
    return winner, fold_df


def run_h005(stack: dict, is_panel: pd.DataFrame, folds: list, s1: pd.Series) -> dict:
    ols, _ = screen_days(
        stack, is_panel, folds, s1, grid=OLS_DAY_GRID, kind="ols",
        ols_days=DAY_OLS_WINDOW, adf_days=DAY_OLS_WINDOW, z_days=DAY_Z_WINDOW,
    )
    stack["OLS_WINDOW_STAR"] = ols
    adf, _ = screen_days(
        stack, is_panel, folds, s1, grid=ADF_DAY_GRID, kind="adf",
        ols_days=ols, adf_days=DAY_OLS_WINDOW, z_days=DAY_Z_WINDOW,
    )
    stack["ADF_WINDOW_STAR"] = adf
    bar = str(stack["BAR_STAR"])
    lb = lookbacks_for_bar(bar, ols_days=ols, z_days=DAY_Z_WINDOW, adf_days=adf)
    rows = []
    for ez in ENTRY_Z_GRID:
        cfg = config_from_stack(
            stack,
            entry_z=ez,
            ols_window=lb["ols_window"],
            z_window=lb["z_window"],
            adf_window=lb["adf_window"],
        )
        part = fold_val_metrics(is_panel, folds, {str(ez): cfg}, s1_weekly=s1)
        rows.append(part)
    fold_ez = pd.concat(rows, ignore_index=True)
    entry_z = float(pick_arm(fold_ez))
    stack["ENTRY_Z_STAR"] = entry_z
    lb = lookbacks_for_bar(bar, ols_days=ols, z_days=DAY_Z_WINDOW, adf_days=adf)
    rows = []
    for zd in Z_DAY_GRID:
        lbz = lookbacks_for_bar(bar, ols_days=ols, z_days=zd, adf_days=adf)
        p = overlay_ols_hedge(
            is_panel,
            ols_window=lbz["ols_window"],
            z_window=lbz["z_window"],
            hl_window=lbz["hl_window"],
            adf_window=lbz["adf_window"],
        )
        cfg = config_from_stack(
            stack,
            entry_z=entry_z,
            ols_window=lbz["ols_window"],
            z_window=lbz["z_window"],
            adf_window=lbz["adf_window"],
        )
        part = fold_val_metrics(p, folds, {str(zd): cfg}, s1_weekly=s1)
        rows.append(part)
    fold_z = pd.concat(rows, ignore_index=True)
    z_win = int(pick_arm(fold_z))
    stack["Z_WINDOW_STAR"] = z_win
    register_hypothesis_arms(
        "H-005",
        [str(d) for d in OLS_DAY_GRID]
        + [str(d) for d in ADF_DAY_GRID]
        + [str(e) for e in ENTRY_Z_GRID]
        + [str(z) for z in Z_DAY_GRID],
        overwrite=True,
    )
    return {
        "OLS_WINDOW_STAR": ols,
        "ADF_WINDOW_STAR": adf,
        "ENTRY_Z_STAR": entry_z,
        "Z_WINDOW_STAR": z_win,
    }


def run_h006(stack: dict, is_panel: pd.DataFrame, folds: list, s1: pd.Series) -> dict:
    bar = str(stack["BAR_STAR"])
    lb = lookbacks_for_bar(
        bar,
        ols_days=int(stack["OLS_WINDOW_STAR"]),
        z_days=int(stack["Z_WINDOW_STAR"]),
        adf_days=int(stack["ADF_WINDOW_STAR"]),
    )
    cfg_ols = config_from_stack(stack, hedge="ols")
    parts = [fold_val_metrics(is_panel, folds, {"ols": cfg_ols}, hyp_id="H-006", s1_weekly=s1)]
    cfg_kf = config_from_stack(stack, hedge="kalman")
    for d in KALMAN_DELTA_GRID:
        p = overlay_kalman_hedge(
            is_panel.copy(),
            burn_in=lb["kalman_burn_in"],
            z_window=lb["z_window"],
            hl_window=lb["hl_window"],
            adf_window=lb["adf_window"],
            delta=d,
        )
        arm = f"kalman_{d:g}"
        parts.append(fold_val_metrics(p, folds, {arm: cfg_kf}, hyp_id="H-006", s1_weekly=s1))
    fold_df = pd.concat(parts, ignore_index=True)
    register_hypothesis_arms(
        "H-006",
        ["ols"] + [f"kalman_{d:g}" for d in KALMAN_DELTA_GRID],
        overwrite=True,
    )
    winner = pick_arm(fold_df)
    out: dict = {"HEDGE_STAR": "ols", "KALMAN_DELTA_STAR": None}
    if winner.startswith("kalman"):
        out["HEDGE_STAR"] = "kalman"
        out["KALMAN_DELTA_STAR"] = float(winner.split("_", 1)[1])
    return out


def run_simple(
    stack: dict,
    is_panel: pd.DataFrame,
    folds: list,
    s1: pd.Series,
    hyp_id: str,
    configs: dict[str, object],
) -> tuple[str, pd.DataFrame]:
    cfgs = {name: config_from_stack(stack, **overrides) for name, overrides in configs.items()}
    fold_df = fold_val_metrics(is_panel, folds, cfgs, hyp_id=hyp_id, s1_weekly=s1)
    register_hypothesis_arms(hyp_id, list(cfgs.keys()), overwrite=True)
    return pick_arm(fold_df), fold_df


def main() -> None:
    baseline = load_star_stack(DEFAULT_STAR_STACK)
    stack = dict(baseline)
    report: dict = {"baseline": baseline, "selections": {}, "changes": {}}

    s1 = load_s1_weekly(ROOT)

    break_star, fold_h003 = run_h003(stack, s1)
    stack["BREAK_STAR"] = break_star
    report["selections"]["H-003"] = {
        "BREAK_STAR": break_star,
        "hint": median_sharpe_hint(fold_h003),
        "table": _agg(fold_h003).to_dict(),
    }

    book_star, pairs_star, fold_h004 = run_h004(stack, s1)
    stack["BOOK_STAR"] = book_star
    stack["PAIRS_STAR"] = pairs_star
    report["selections"]["H-004"] = {
        "BOOK_STAR": book_star,
        "PAIRS_STAR": pairs_star,
        "table": _agg(fold_h004).to_dict(),
    }

    is_panel, _, s1 = prepare_panel(stack, use_stack_windows=True)
    folds = make_folds(is_panel, str(stack["BAR_STAR"]))

    h005 = run_h005(stack, is_panel, folds, s1)
    for k, v in h005.items():
        stack[k] = v
    report["selections"]["H-005"] = h005

    is_panel, _, s1 = prepare_panel(stack, use_stack_windows=True)
    folds = make_folds(is_panel, str(stack["BAR_STAR"]))
    h006 = run_h006(stack, is_panel, folds, s1)
    stack.update(h006)
    report["selections"]["H-006"] = h006

    is_panel, _, s1 = prepare_panel(stack, use_stack_windows=True)
    folds = make_folds(is_panel, str(stack["BAR_STAR"]))

    trend, fold_h007 = run_simple(
        stack, is_panel, folds, s1, "H-007",
        {
            "off": {"trend_mode": "off"},
            "adx_veto": {"trend_mode": "adx_veto"},
            "rsi_confirm": {"trend_mode": "rsi_confirm"},
            "both": {"trend_mode": "both"},
        },
    )
    stack["TREND_STAR"] = trend
    report["selections"]["H-007"] = {"TREND_STAR": trend, "hint": median_sharpe_hint(fold_h007)}

    hl, fold_h008 = run_simple(
        stack, is_panel, folds, s1, "H-008",
        {
            "off": {"hl_gate_min": None, "hl_gate_max": None},
            "10_60": {"hl_gate_min": 10.0, "hl_gate_max": 60.0},
            "5_30": {"hl_gate_min": 5.0, "hl_gate_max": 30.0},
        },
    )
    stack["HL_GATE_STAR"] = "off" if hl == "off" else hl
    report["selections"]["H-008"] = {"HL_GATE_STAR": hl, "hint": median_sharpe_hint(fold_h008)}

    overlap, _ = run_simple(
        stack, is_panel, folds, s1, "H-009",
        {"allow": {"overlap_mode": "allow"}, "never_allow": {"overlap_mode": "never_allow"}},
    )
    stack["OVERLAP_STAR"] = overlap
    report["selections"]["H-009"] = {"OVERLAP_STAR": overlap}

    exit_star, _ = run_simple(
        stack, is_panel, folds, s1, "H-010",
        {"mean_only": {"exit_mode": "mean_only"}, "hl3_atr_breaker": {"exit_mode": "hl3_atr_breaker"}},
    )
    stack["EXIT_STAR"] = exit_star
    report["selections"]["H-010"] = {"EXIT_STAR": exit_star}

    corr, _ = run_simple(
        stack, is_panel, folds, s1, "H-011",
        {
            "off": {"corr_k": None},
            "k0.50": {"corr_k": 0.50},
            "k0.70": {"corr_k": 0.70},
            "k0.90": {"corr_k": 0.90},
        },
    )
    stack["CORR_GATE_STAR"] = "off" if corr == "off" else corr
    report["selections"]["H-011"] = {"CORR_GATE_STAR": corr}

    size, _ = run_simple(
        stack, is_panel, folds, s1, "H-012",
        {
            "equal": {"size_mode": "equal"},
            "score": {"size_mode": "score"},
            "score_conf": {"size_mode": "score_conf"},
        },
    )
    stack["SIZE_STAR"] = size
    report["selections"]["H-012"] = {"SIZE_STAR": size}

    vol, _ = run_simple(
        stack, is_panel, folds, s1, "H-013",
        {
            "fixed_k": {"vol_mode": "fixed_k"},
            "kt": {"vol_mode": "kt"},
            "s1_vt": {"vol_mode": "s1_vt"},
        },
    )
    stack["VOL_STAR"] = vol
    report["selections"]["H-013"] = {"VOL_STAR": vol}

    zmode, _ = run_simple(
        stack, is_panel, folds, s1, "H-014",
        {
            "fixed": {"z_window_mode": "fixed"},
            "adaptive": {"z_window_mode": "adaptive"},
            "adaptive_alt": {"z_window_mode": "adaptive_alt"},
        },
    )
    stack["Z_WINDOW_MODE_STAR"] = zmode
    report["selections"]["H-014"] = {"Z_WINDOW_MODE_STAR": zmode}

    entry, _ = run_simple(
        stack, is_panel, folds, s1, "H-015",
        {
            "trad_z": {"entry_mode": "trad_z"},
            "v1_roll_asym": {"entry_mode": "v1_roll_asym", "k_in": 2.0, "k_out": 0.5},
            "v1_ewm_asym": {"entry_mode": "v1_ewm_asym", "k_in": 2.0, "k_out": 0.5},
            "v2_ou": {"entry_mode": "v2_ou", "k_in": 2.0, "k_out": 0.5},
            "v3_hmm_innov": {"entry_mode": "v3_hmm_innov"},
        },
    )
    stack["ENTRY_STAR"] = entry
    report["selections"]["H-015"] = {"ENTRY_STAR": entry}

    star_keys = [
        "BREAK_STAR", "BOOK_STAR", "PAIRS_STAR",
        "OLS_WINDOW_STAR", "ADF_WINDOW_STAR", "ENTRY_Z_STAR", "Z_WINDOW_STAR",
        "HEDGE_STAR", "KALMAN_DELTA_STAR",
        "TREND_STAR", "HL_GATE_STAR", "OVERLAP_STAR", "EXIT_STAR",
        "CORR_GATE_STAR", "SIZE_STAR", "VOL_STAR", "Z_WINDOW_MODE_STAR", "ENTRY_STAR",
    ]
    for key in star_keys:
        old = baseline.get(key)
        new = stack.get(key)
        if old != new:
            report["changes"][key] = {"from": old, "to": new}

    out_path = os.path.join(os.path.dirname(__file__), "artifacts", "s2_star_auto_review.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
        f.write("\n")

    print("=== STAR auto-review (H-003..H-015) ===")
    print(json.dumps(report["changes"], indent=2, default=str))
    print(f"written {out_path}")
    if report["changes"]:
        save_star_stack(DEFAULT_STAR_STACK, stack)
        print("updated", DEFAULT_STAR_STACK)


if __name__ == "__main__":
    main()
