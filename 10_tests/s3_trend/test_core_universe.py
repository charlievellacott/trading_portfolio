"""S3 core universe registry and the shared research cut."""

from __future__ import annotations

import pandas as pd

from data.processing.s3_fx_price_panel import (
    RESEARCH_IS_END_S3,
    S3_HOLDOUT_END,
    core_universe,
)


def test_shared_research_cut():
    assert RESEARCH_IS_END_S3 == pd.Timestamp("2016-12-31")
    assert S3_HOLDOUT_END == pd.Timestamp("2019-12-31")


def test_core_universes():
    fx = core_universe("A")
    etf = core_universe("B")
    assert fx["kind"] == "fx"
    assert etf["kind"] == "etf"
    assert "SPY" in etf["instruments"]
    assert "EURUSD" in fx["instruments"]
