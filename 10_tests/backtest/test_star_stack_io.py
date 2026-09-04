"""Star-stack JSON I/O (sleeve-agnostic)."""

from __future__ import annotations

import os

import pytest

from backtest.star_stack_io import (
    default_star_stack_path,
    load_star_stack,
    require_star,
    save_star_stack,
    update_star_stack_key,
    vt_target_ann_vol_from_stack,
)
from data.repo_paths import repo_root


def test_round_trip_load_save(tmp_path):
    path = os.path.join(str(tmp_path), "stack.json")
    payload = {"N_STAR": 15, "VT_TARGET_ANN_VOL_STAR": 0.10}
    save_star_stack(path, payload)
    loaded = load_star_stack(path)
    assert loaded["N_STAR"] == 15
    assert loaded["VT_TARGET_ANN_VOL_STAR"] == pytest.approx(0.10)


def test_load_missing_raises(tmp_path):
    path = os.path.join(str(tmp_path), "missing.json")
    with pytest.raises(FileNotFoundError, match="star stack not found"):
        load_star_stack(path)


def test_require_star_raises_on_none():
    with pytest.raises(ValueError, match="FOO_STAR"):
        require_star("FOO_STAR", None)
    require_star("FOO_STAR", "ok")


def test_update_star_stack_key_merges(tmp_path):
    path = os.path.join(str(tmp_path), "stack.json")
    save_star_stack(path, {"N_STAR": 15, "KEEP": 1})
    out = update_star_stack_key(
        path,
        "VT_TARGET_ANN_VOL_STAR",
        0.06,
        extra={"VT_TARGET_PICK_STAR": "calmar"},
    )
    assert out["N_STAR"] == 15
    assert out["KEEP"] == 1
    assert out["VT_TARGET_ANN_VOL_STAR"] == pytest.approx(0.06)
    assert out["VT_TARGET_PICK_STAR"] == "calmar"
    again = load_star_stack(path)
    assert again["KEEP"] == 1


def test_vt_target_ann_vol_from_stack_default_and_override():
    assert vt_target_ann_vol_from_stack({}) == pytest.approx(0.10)
    assert vt_target_ann_vol_from_stack({}, default=0.12) == pytest.approx(0.12)
    assert vt_target_ann_vol_from_stack(
        {"VT_TARGET_ANN_VOL_STAR": 0.06}
    ) == pytest.approx(0.06)
    assert vt_target_ann_vol_from_stack(
        {"vt_target_ann_vol": 0.08}
    ) == pytest.approx(0.08)


def test_default_star_stack_path_sleeves():
    root = repo_root()
    s1 = default_star_stack_path("s1", repo_root_dir=root)
    s2 = default_star_stack_path("s2", repo_root_dir=root)
    assert s1.endswith(os.path.join("s1_equities", "artifacts", "s1_star_stack.json"))
    assert s2.endswith(os.path.join("s2_coint", "artifacts", "s2_star_stack.json"))
    with pytest.raises(ValueError, match="unknown sleeve"):
        default_star_stack_path("fx")
