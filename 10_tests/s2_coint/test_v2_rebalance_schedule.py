"""Tests for the v2 Pass B rotate-schedule builder (1d and 1h)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.s2_coint.rotation import (
    V2_REBALANCE_COLS,
    build_v2_rebalance_schedule,
    quarter_end_rebalance_dates,
)


def _hourly_index(n_sessions: int, bars_per_session: int = 6) -> pd.DatetimeIndex:
    days = pd.bdate_range("2025-01-02", periods=n_sessions)
    stamps = [
        d + pd.Timedelta(hours=14, minutes=30) + pd.Timedelta(hours=h)
        for d in days
        for h in range(bars_per_session)
    ]
    return pd.DatetimeIndex(stamps)


def _coint_closes(idx: pd.DatetimeIndex, seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    n = len(idx)
    x = 4.0 + np.cumsum(rng.normal(0, 0.01, n))
    y = 0.5 + 1.2 * x + rng.normal(0, 0.005, n)
    z = 4.0 + np.cumsum(rng.normal(0, 0.01, n))
    return {
        "AAA": pd.Series(np.exp(y), index=idx),
        "BBB": pd.Series(np.exp(x), index=idx),
        "CCC": pd.Series(np.exp(z), index=idx),
    }


def test_hourly_schedule_schema_effective_next_bar_and_full_window():
    bps = 6
    idx = _hourly_index(140, bps)
    closes = _coint_closes(idx)
    rebals = quarter_end_rebalance_dates(idx)
    L = 60
    reb = build_v2_rebalance_schedule(
        closes,
        [("AAA", "BBB"), ("AAA", "CCC")],
        idx,
        {"quarterly": rebals},
        [L],
        pool="sector",
        fdr="raw_p05",
        ols_window=L * bps,
        bars_per_session=bps,
        min_is_bars=L * bps,
    )
    assert tuple(reb.columns) == V2_REBALANCE_COLS
    assert not reb.empty
    assert set(reb["schedule"]) == {"quarterly"}
    assert set(reb["lookback"]) == {L}
    assert "AAA|BBB" in set(reb["pair_id"]) or "BBB|AAA" in set(reb["pair_id"])
    for row in reb.itertuples(index=False):
        later = idx[idx > row.rebalance_date]
        assert row.effective_date == later[0]
        assert row.effective_date > row.rebalance_date
    # Rebalances with fewer than L sessions of history are skipped (full window required).
    first_ok = idx[L * bps - 1]
    assert (reb["rebalance_date"] >= first_ok).all()


def test_rebalance_without_later_bar_is_dropped_and_empty_keeps_schema():
    idx = _hourly_index(80, 6)
    closes = _coint_closes(idx, seed=1)
    reb = build_v2_rebalance_schedule(
        closes,
        [("AAA", "BBB")],
        idx,
        {"12m6m": [idx[-1]]},
        [252],
        pool="sector",
        fdr="bh_fdr",
        ols_window=252 * 6,
        bars_per_session=6,
        min_is_bars=252 * 6,
    )
    assert reb.empty
    assert tuple(reb.columns) == V2_REBALANCE_COLS
