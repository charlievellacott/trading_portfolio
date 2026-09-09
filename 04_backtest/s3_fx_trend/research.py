"""Notebook helpers for the S3 FX trend hypothesis stack (research runner)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import pandas as pd

from backtest.star_stack_io import (
    VT_TARGET_ANN_VOL_STAR,
    load_star_stack,
    require_star,
    vt_target_ann_vol_from_stack,
)
from data.processing.s3_fx_price_panel import RESEARCH_IS_END_S3
from strategies.s3_fx_trend.config import S3SimConfig

_HERE = os.path.abspath(__file__)
_PKG_DIR = os.path.dirname(_HERE)


def repo_root(start: str | None = None) -> str:
    cur = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isdir(os.path.join(cur, "01_data", "ingestion")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start or os.getcwd())
        cur = parent


# Strict editable installs resolve __file__ under build/; STAR JSON lives in the repo.
ARTIFACTS_DIR = os.path.join(
    repo_root(_PKG_DIR), "04_backtest", "s3_fx_trend", "artifacts"
)
DEFAULT_STAR_STACK_CORE = os.path.join(ARTIFACTS_DIR, "s3_core_star_stack.json")
DEFAULT_STAR_STACK_EVENT = os.path.join(ARTIFACTS_DIR, "s3_event_star_stack.json")
DEFAULT_VARIANT_LEDGER_CORE = os.path.join(ARTIFACTS_DIR, "s3_variant_ledger_core.json")
DEFAULT_VARIANT_LEDGER_EVENT = os.path.join(
    ARTIFACTS_DIR, "s3_variant_ledger_event.json"
)

HYPOTHESIS_ORDER_CORE: tuple[str, ...] = tuple(f"H-{i:03d}" for i in range(1, 14))
HYPOTHESIS_ORDER_EVENT: tuple[str, ...] = tuple(f"E-{i:03d}" for i in range(1, 7))

_BLEND_FROM_HORIZON = {
    "1M": "single_1m",
    "1m": "single_1m",
    "single_1m": "single_1m",
    "3M": "single_3m",
    "3m": "single_3m",
    "single_3m": "single_3m",
    "12M": "single_12m",
    "12m": "single_12m",
    "single_12m": "single_12m",
    "equal_blend": "equal_blend",
    "blend": "equal_blend",
}

_FAMILY_MAP = {
    "tsmom": "tsmom",
    "TSMOM": "tsmom",
    "donchian": "donchian",
    "Donchian": "donchian",
    "classic_donchian": "donchian",
    "both": "both",
}

_CONVICTION_MAP = {
    "sign": "sign",
    "sign_only": "sign",
    "abs": "abs",
    "abs_scale": "abs",
}

_DONCHIAN_HL_MAP = {
    "daily_ny_close": "daily_ny_close",
    "ny_close": "daily_ny_close",
    "overlap_confirmed": "overlap_confirmed",
    "ex_asia_eurgbp": "ex_asia_eurgbp",
}


def _normalize_sleeve(sleeve: str) -> str:
    key = str(sleeve).strip().lower()
    if key in {"core", "s3_core", "s3", "s3_fx_trend"}:
        return "core"
    if key in {"event", "s3_event"}:
        return "event"
    raise ValueError(f"sleeve must be 'core' or 'event', got {sleeve!r}")


def variant_ledger_path(sleeve: str = "core") -> str:
    key = _normalize_sleeve(sleeve)
    if key == "event":
        return DEFAULT_VARIANT_LEDGER_EVENT
    return DEFAULT_VARIANT_LEDGER_CORE


def star_stack_path(sleeve: str = "core") -> str:
    key = _normalize_sleeve(sleeve)
    if key == "event":
        return DEFAULT_STAR_STACK_EVENT
    return DEFAULT_STAR_STACK_CORE


def hypothesis_order(sleeve: str = "core") -> tuple[str, ...]:
    key = _normalize_sleeve(sleeve)
    if key == "event":
        return HYPOTHESIS_ORDER_EVENT
    return HYPOTHESIS_ORDER_CORE


def require_stars(stack: dict, keys: Sequence[str]) -> None:
    """Raise if any named STAR key is missing / None."""
    for key in keys:
        require_star(str(key), stack.get(key))


def split_is_oos(
    panel: pd.DataFrame,
    *,
    is_end: pd.Timestamp | str = RESEARCH_IS_END_S3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on calendar day: bars on ``is_end`` stay in IS (incl. intraday)."""
    d = pd.to_datetime(panel["date"]).dt.normalize()
    end = pd.Timestamp(is_end).normalize()
    return panel.loc[d <= end].copy(), panel.loc[d > end].copy()


def _as_tuple(raw) -> tuple:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return (raw,)


def _map_tsmom_blend(stack: dict) -> str | None:
    raw = stack.get("TSMOM_BLEND_STAR")
    if raw in (None, "None", ""):
        raw = stack.get("HORIZON_STAR")
    if raw in (None, "None", ""):
        return None
    mapped = _BLEND_FROM_HORIZON.get(str(raw), str(raw))
    return mapped


def _map_signal_family(stack: dict) -> str | None:
    raw = stack.get("SIGNAL_FAMILY_STAR")
    if raw in (None, "None", ""):
        raw = stack.get("FAMILY_STAR")
    if raw in (None, "None", ""):
        return None
    return _FAMILY_MAP.get(str(raw), str(raw))


def _overlay_modes(stack: dict) -> dict[str, str]:
    """Resolve carry/value modes from OVERLAY_STAR when explicit keys unset."""
    out: dict[str, str] = {}
    carry = stack.get("CARRY_MODE_STAR")
    value = stack.get("VALUE_MODE_STAR")
    overlay = stack.get("OVERLAY_STAR")
    if carry not in (None, "None", ""):
        out["carry_mode"] = str(carry)
    if value not in (None, "None", ""):
        out["value_mode"] = str(value)
    if overlay in (None, "None", "", "none", "off"):
        return out
    ov = str(overlay).lower()
    if "carry_mode" not in out:
        if ov in {"carry_gate", "gate_carry"}:
            out["carry_mode"] = "gate"
        elif ov in {"carry_tilt", "tilt_carry"}:
            out["carry_mode"] = "tilt"
    if "value_mode" not in out:
        if ov in {"value_gate", "gate_value"}:
            out["value_mode"] = "gate"
        elif ov in {"value_tilt", "tilt_value"}:
            out["value_mode"] = "tilt"
    return out


def config_from_stack(
    stack: dict,
    hyp_overrides: dict | None = None,
) -> S3SimConfig:
    """Map frozen ``*_STAR`` keys onto ``S3SimConfig`` (unset → dataclass defaults)."""
    kwargs: dict = {}

    pairs = stack.get("PAIRS_STAR")
    if pairs not in (None, "None", ""):
        kwargs["pairs"] = tuple(str(p) for p in _as_tuple(pairs))

    bar = stack.get("BAR_STAR")
    if bar not in (None, "None", ""):
        kwargs["bar"] = str(bar)

    family = _map_signal_family(stack)
    if family is not None:
        kwargs["signal_family"] = family

    lookbacks = stack.get("TSMOM_LOOKBACKS_STAR")
    if lookbacks not in (None, "None", ""):
        kwargs["tsmom_lookbacks"] = tuple(int(x) for x in _as_tuple(lookbacks))

    blend = _map_tsmom_blend(stack)
    if blend is not None:
        kwargs["tsmom_blend"] = blend

    donch_n = stack.get("DONCHIAN_N_STAR")
    if donch_n not in (None, "None", ""):
        kwargs["donchian_n"] = int(donch_n)

    hl_src = stack.get("DONCHIAN_HL_SOURCE_STAR")
    if hl_src in (None, "None", ""):
        hl_src = stack.get("BREAKOUT_DEF_STAR")
    if hl_src not in (None, "None", ""):
        kwargs["donchian_hl_source"] = _DONCHIAN_HL_MAP.get(str(hl_src), str(hl_src))

    if stack.get("PAIR_VT_STAR") not in (None, "None", ""):
        kwargs["pair_vt"] = bool(stack.get("PAIR_VT_STAR"))
    if stack.get("BOOK_VT_STAR") not in (None, "None", ""):
        kwargs["book_vt"] = bool(stack.get("BOOK_VT_STAR"))

    # Composite RISK_STAR may encode pair/book VT as a string label.
    risk = stack.get("RISK_STAR")
    if risk not in (None, "None", "") and "pair_vt" not in kwargs:
        r = str(risk).lower()
        if r in {"pair_vt", "pair_only"}:
            kwargs["pair_vt"] = True
        elif r in {"pair_plus_book_vt", "book_vt", "both_vt"}:
            kwargs["pair_vt"] = True
            kwargs["book_vt"] = True
        elif r in {"off", "raw", "none"}:
            kwargs["pair_vt"] = False
            kwargs["book_vt"] = False

    kwargs["vt_target_ann_vol"] = vt_target_ann_vol_from_stack(stack)

    conv = stack.get("CONVICTION_SCALING_STAR")
    if conv in (None, "None", ""):
        conv = stack.get("CONVICTION_STAR")
    if conv not in (None, "None", ""):
        kwargs["conviction_scaling"] = _CONVICTION_MAP.get(str(conv), str(conv))

    if stack.get("WEAK_SIGNAL_FLAT_Z_STAR") not in (None, "None", ""):
        kwargs["weak_signal_flat_z"] = float(stack["WEAK_SIGNAL_FLAT_Z_STAR"])
    if stack.get("REBALANCE_BAND_STAR") not in (None, "None", ""):
        kwargs["rebalance_band"] = float(stack["REBALANCE_BAND_STAR"])

    kwargs.update(_overlay_modes(stack))

    for star_key, field in (
        ("CARRY_REFRESH_STAR", "carry_refresh"),
        ("VALUE_REFRESH_STAR", "value_refresh"),
    ):
        raw = stack.get(star_key)
        if raw not in (None, "None", ""):
            kwargs[field] = str(raw)

    for star_key, field in (
        ("CARRY_AGREE_SCALE_STAR", "carry_agree_scale"),
        ("CARRY_DISAGREE_SCALE_STAR", "carry_disagree_scale"),
        ("VALUE_AGREE_SCALE_STAR", "value_agree_scale"),
        ("VALUE_DISAGREE_SCALE_STAR", "value_disagree_scale"),
        ("SPIKE_THRESHOLD_SIGMAS_STAR", "spike_threshold_sigmas"),
        ("CORR_GATE_THRESHOLD_STAR", "corr_gate_threshold"),
        ("CURRENCY_CAP_GROSS_PCT_STAR", "currency_cap_gross_pct"),
        ("FINANCING_SPREAD_BPS_ANNUAL_STAR", "financing_spread_bps_annual"),
        ("EVENT_Z_GATE_STAR", "event_z_gate"),
    ):
        raw = stack.get(star_key)
        if raw not in (None, "None", ""):
            kwargs[field] = float(raw)

    # Event gate alias from hyp log.
    if "event_z_gate" not in kwargs and stack.get("K_STAR") not in (None, "None", ""):
        kwargs["event_z_gate"] = float(stack["K_STAR"])

    for star_key, field in (
        ("SPIKE_MODE_STAR", "spike_mode"),
        ("CORR_CONFLICT_MODE_STAR", "corr_conflict_mode"),
        ("COST_PROFILE_STAR", "cost_profile"),
        ("EVENT_BAR_STAR", "event_bar"),
    ):
        raw = stack.get(star_key)
        if raw not in (None, "None", ""):
            kwargs[field] = str(raw)

    if stack.get("INCLUDE_SWAP_STAR") not in (None, "None", ""):
        kwargs["include_swap"] = bool(stack["INCLUDE_SWAP_STAR"])
    if stack.get("EVENT_IMPACT_MIN_STAR") not in (None, "None", ""):
        kwargs["event_impact_min"] = int(stack["EVENT_IMPACT_MIN_STAR"])
    if stack.get("EVENT_HOLD_BARS_STAR") not in (None, "None", ""):
        kwargs["event_hold_bars"] = int(stack["EVENT_HOLD_BARS_STAR"])
    if stack.get("SURPRISE_SIZING_STAR") not in (None, "None", ""):
        kwargs["surprise_sizing"] = bool(stack["SURPRISE_SIZING_STAR"])

    if hyp_overrides:
        kwargs.update(hyp_overrides)
    # Drop VT default overwrite if neither stack nor override set it explicitly.
    if (
        VT_TARGET_ANN_VOL_STAR not in stack
        and "vt_target_ann_vol" not in (hyp_overrides or {})
        and stack.get("vt_target_ann_vol") is None
    ):
        # Keep dataclass default via from_dict only when key was injected by helper default.
        # vt_target_ann_vol_from_stack always returns a float default — fine to pass.
        pass

    return S3SimConfig.from_dict(kwargs)


def load_variant_ledger(path: str | None = None, *, sleeve: str = "core") -> dict:
    p = path or variant_ledger_path(sleeve)
    if not os.path.isfile(p):
        return {"entries": [], "cumulative_arms": 0}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_variant_ledger(
    payload: dict,
    path: str | None = None,
    *,
    sleeve: str = "core",
) -> None:
    p = path or variant_ledger_path(sleeve)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    entries = payload.get("entries") or []
    payload["cumulative_arms"] = int(
        sum(int(e.get("n_arms", len(e.get("arms", [])))) for e in entries)
    )
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def _normalize_hyp_id(hyp_id: str, *, sleeve: str = "core") -> str:
    key = str(hyp_id).strip().upper()
    order = hypothesis_order(sleeve)
    if key not in order:
        raise KeyError(f"unknown hyp_id {hyp_id!r}; expected one of {order}")
    return key


def cumulative_trials_before(
    hyp_id: str,
    *,
    sleeve: str = "core",
    path: str | None = None,
) -> int:
    key = _normalize_hyp_id(hyp_id, sleeve=sleeve)
    order = hypothesis_order(sleeve)
    idx = order.index(key)
    if idx == 0:
        return 0
    prior = set(order[:idx])
    ledger = load_variant_ledger(path, sleeve=sleeve)
    total = 0
    for entry in ledger.get("entries") or []:
        eid = str(entry.get("hyp_id", "")).strip().upper()
        if eid in prior:
            total += int(entry.get("n_arms", len(entry.get("arms", []))))
    return total


def n_trials_local(configs: dict) -> int:
    return len(configs)


def n_trials_ledger_total(*, sleeve: str = "core", path: str | None = None) -> int:
    ledger = load_variant_ledger(path, sleeve=sleeve)
    total = 0
    for entry in ledger.get("entries") or []:
        total += int(entry.get("n_arms", len(entry.get("arms", []))))
    return int(total)


def n_trials_stack(
    hyp_id: str,
    configs: dict,
    *,
    sleeve: str = "core",
    path: str | None = None,
) -> int:
    return cumulative_trials_before(hyp_id, sleeve=sleeve, path=path) + n_trials_local(
        configs
    )


def register_hypothesis_arms(
    hyp_id: str,
    arm_names: Sequence[str],
    *,
    sleeve: str = "core",
    overwrite: bool = False,
    path: str | None = None,
) -> dict:
    """Append or replace the ledger entry for a hypothesis screen."""
    key = _normalize_hyp_id(hyp_id, sleeve=sleeve)
    arms = [str(a) for a in arm_names]
    ledger = load_variant_ledger(path, sleeve=sleeve)
    entries: list[dict] = list(ledger.get("entries") or [])
    row = {"hyp_id": key, "arms": arms, "n_arms": len(arms)}
    replaced = False
    for i, entry in enumerate(entries):
        if str(entry.get("hyp_id", "")).strip().upper() == key:
            if not overwrite:
                raise ValueError(
                    f"{key} already in variant ledger; pass overwrite=True to replace"
                )
            entries[i] = row
            replaced = True
            break
    if not replaced:
        entries.append(row)
    order = hypothesis_order(sleeve)
    entries.sort(
        key=lambda e: order.index(_normalize_hyp_id(e["hyp_id"], sleeve=sleeve))
    )
    ledger["entries"] = entries
    save_variant_ledger(ledger, path, sleeve=sleeve)
    return ledger
