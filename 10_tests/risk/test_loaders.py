"""Sealed-return loader failures (no network)."""

from __future__ import annotations

import os

import pytest

from risk.leverage.loaders import load_s2_period_returns_base
from risk.monte_carlo.loaders import find_repo_root, load_s2_period_returns, require_parquet


def test_find_repo_root():
    root = find_repo_root(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.isfile(os.path.join(root, "pyproject.toml"))
    assert os.path.isdir(os.path.join(root, "06_risk"))


def test_missing_parquet_is_loud(tmp_path):
    missing = os.path.join(str(tmp_path), "nope.parquet")
    with pytest.raises(FileNotFoundError, match="Sealed period returns missing"):
        require_parquet(missing, "export hint")
    with pytest.raises(FileNotFoundError, match="Sealed period returns missing"):
        load_s2_period_returns(missing)


def test_missing_base_parquet_is_loud(tmp_path):
    missing_root = str(tmp_path)
    os.makedirs(os.path.join(missing_root, "06_risk"), exist_ok=True)
    with open(os.path.join(missing_root, "pyproject.toml"), "w", encoding="utf-8") as f:
        f.write("[project]\nname='t'\n")
    with pytest.raises(FileNotFoundError, match="Unlevered base period returns missing"):
        load_s2_period_returns_base(missing_root)
