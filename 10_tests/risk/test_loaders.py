"""Sealed-return loader failures (no network)."""

from __future__ import annotations

import os

import pytest

from risk.analytics.leverage.loaders import (
    load_s2_period_returns_base,
    s2_period_returns_base_path,
)
from risk.analytics.monte_carlo.loaders import (
    find_repo_root,
    load_s2_period_returns,
    require_parquet,
    s2_period_returns_path,
)


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


def test_s2_base_path_is_not_sealed_net():
    root = find_repo_root(os.path.dirname(os.path.abspath(__file__)))
    base = s2_period_returns_base_path(root)
    sealed = s2_period_returns_path(root)
    assert base.endswith(os.path.join("s2_coint", "s2_period_returns_base.parquet"))
    assert sealed.endswith(os.path.join("s2_coint", "s2_period_returns.parquet"))
    assert base != sealed


def test_missing_base_parquet_is_loud(tmp_path):
    missing_root = str(tmp_path)
    os.makedirs(os.path.join(missing_root, "06_risk"), exist_ok=True)
    with open(os.path.join(missing_root, "pyproject.toml"), "w", encoding="utf-8") as f:
        f.write("[project]\nname='t'\n")
    with pytest.raises(FileNotFoundError, match="Unlevered base period returns missing"):
        load_s2_period_returns_base(missing_root)
