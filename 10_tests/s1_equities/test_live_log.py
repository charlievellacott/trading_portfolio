"""Tests for live_log ledger and metrics live bridges."""

from __future__ import annotations

import os
import tempfile

import pandas as pd

from performance.live_log import (
    STRATEGY_S1_EQUITIES,
    get_meta,
    log_equity_daily,
    log_fills,
    log_signals,
    portfolio_equity_series,
    set_portfolio_go_live,
    snapshot_mark_to_market,
    strategy_equity_series,
    sync_stop_fills,
)
from performance.metrics import (
    apply_go_live_start,
    equity_to_returns,
    live_portfolio_metrics,
    sharpe_ratio,
    summary_metrics_table,
)


class _FakeAccount:
    def __init__(self, equity: float, cash: float = 0.0) -> None:
        self.equity = equity
        self.cash = cash


class _FakeBroker:
    def __init__(self, equity: float, positions=None, filled_orders=None) -> None:
        self._equity = equity
        self._positions = list(positions or [])
        self._filled_orders = list(filled_orders or [])

    def get_account(self):
        return _FakeAccount(self._equity, cash=self._equity * 0.1)

    def get_positions(self):
        return []

    def get_positions_normalized(self):
        return list(self._positions)

    def get_filled_orders(self, after=None, until=None):
        rows = list(self._filled_orders)
        if after is not None:
            after_ts = pd.Timestamp(after)
            rows = [
                r
                for r in rows
                if r.get("filled_at") is None
                or pd.Timestamp(r["filled_at"]) >= after_ts
            ]
        return rows


def test_signal_and_fill_upsert_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        weights = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-08")] * 2,
                "ticker": ["AAA", "BBB"],
                "weight": [0.1, -0.1],
                "score": [1.2, -0.5],
                "feature_date": [pd.Timestamp("2024-01-05")] * 2,
            }
        )
        log_signals(STRATEGY_S1_EQUITIES, weights, run_id="run1", cache_dir=tmp)
        log_signals(STRATEGY_S1_EQUITIES, weights, run_id="run2", cache_dir=tmp)

        from performance.live_log import load_signals

        signals = load_signals(cache_dir=tmp)
        assert len(signals) == 2
        assert set(signals["run_id"]) == {"run2"}

        log_fills(
            [
                {
                    "strategy_id": STRATEGY_S1_EQUITIES,
                    "ticker": "AAA",
                    "side": "buy",
                    "qty": 10,
                    "price": 100.0,
                    "order_id": "ord-1",
                    "fill_type": "entry",
                    "run_id": "run1",
                }
            ],
            cache_dir=tmp,
        )
        log_fills(
            [
                {
                    "strategy_id": STRATEGY_S1_EQUITIES,
                    "ticker": "AAA",
                    "side": "buy",
                    "qty": 12,
                    "price": 101.0,
                    "order_id": "ord-1",
                    "fill_type": "entry",
                    "run_id": "run2",
                }
            ],
            cache_dir=tmp,
        )
        from performance.live_log import load_fills

        fills = load_fills(cache_dir=tmp)
        assert len(fills) == 1
        assert float(fills.iloc[0]["qty"]) == 12.0


def test_equity_series_go_live_none_and_set() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_equity_daily(
            [
                {
                    "date": "2024-01-02",
                    "level": "portfolio",
                    "strategy_id": "",
                    "equity": 100.0,
                },
                {
                    "date": "2024-01-03",
                    "level": "portfolio",
                    "strategy_id": "",
                    "equity": 101.0,
                },
                {
                    "date": "2024-01-04",
                    "level": "portfolio",
                    "strategy_id": "",
                    "equity": 102.0,
                },
                {
                    "date": "2024-01-02",
                    "level": "strategy",
                    "strategy_id": STRATEGY_S1_EQUITIES,
                    "equity": 100.0,
                },
                {
                    "date": "2024-01-03",
                    "level": "strategy",
                    "strategy_id": STRATEGY_S1_EQUITIES,
                    "equity": 100.5,
                },
                {
                    "date": "2024-01-04",
                    "level": "strategy",
                    "strategy_id": STRATEGY_S1_EQUITIES,
                    "equity": 101.0,
                },
            ],
            cache_dir=tmp,
        )

        meta = get_meta(cache_dir=tmp)
        assert meta["portfolio_go_live"] is None
        full = portfolio_equity_series(cache_dir=tmp)
        assert len(full) == 3

        set_portfolio_go_live("2024-01-03", cache_dir=tmp)
        clipped = portfolio_equity_series(cache_dir=tmp)
        assert list(clipped.index.date) == [
            pd.Timestamp("2024-01-03").date(),
            pd.Timestamp("2024-01-04").date(),
        ]
        sleeve = strategy_equity_series(STRATEGY_S1_EQUITIES, cache_dir=tmp)
        assert len(sleeve) == 2


def test_apply_go_live_start_helper() -> None:
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    series = pd.Series([1.0, 2.0, 3.0], index=idx)
    assert len(apply_go_live_start(series, None)) == 3
    out = apply_go_live_start(series, "2024-01-02")
    assert list(out.values) == [2.0, 3.0]


def test_metrics_bridge_on_synthetic_equity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dates = pd.bdate_range("2024-01-02", periods=30)
        equity = 100.0
        rows = []
        for i, d in enumerate(dates):
            equity *= 1.001
            rows.append(
                {
                    "date": d,
                    "level": "portfolio",
                    "strategy_id": "",
                    "equity": equity,
                }
            )
            rows.append(
                {
                    "date": d,
                    "level": "strategy",
                    "strategy_id": STRATEGY_S1_EQUITIES,
                    "equity": equity,
                }
            )
        log_equity_daily(rows, cache_dir=tmp)

        port = live_portfolio_metrics(cache_dir=tmp)
        assert port["summary"] is not None
        assert "sharpe" in port["summary"].index
        rets = equity_to_returns(port["equity"])
        assert sharpe_ratio(rets) == float(port["summary"]["sharpe"])
        table = summary_metrics_table(rets)
        assert set(table.index) >= {"sharpe", "sortino", "calmar", "max_drawdown"}


def test_sync_stop_fills_and_mtm_with_fake_broker() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_fills(
            [
                {
                    "ts": "2024-01-08T15:00:00",
                    "strategy_id": STRATEGY_S1_EQUITIES,
                    "ticker": "AAA",
                    "side": "buy",
                    "qty": 10,
                    "price": 100.0,
                    "order_id": "entry-1",
                    "fill_type": "entry",
                    "run_id": "r1",
                },
                {
                    "ts": "2024-01-08T15:01:00",
                    "strategy_id": STRATEGY_S1_EQUITIES,
                    "ticker": "AAA",
                    "side": "sell",
                    "qty": 0.0,
                    "price": 80.0,
                    "order_id": "stop-1",
                    "fill_type": "stop_submitted",
                    "run_id": "r1",
                },
            ],
            cache_dir=tmp,
        )

        broker = _FakeBroker(
            equity=50_000.0,
            positions=[
                {
                    "ticker": "AAA",
                    "qty": 10.0,
                    "avg_entry": 100.0,
                    "market_value": 1050.0,
                    "unrealized_pl": 50.0,
                    "current_price": 105.0,
                }
            ],
            filled_orders=[
                {
                    "order_id": "stop-1",
                    "ticker": "AAA",
                    "side": "sell",
                    "qty": 10.0,
                    "price": 80.0,
                    "filled_at": "2024-01-10T14:30:00",
                    "order_type": "stop",
                }
            ],
        )

        # First MTM while still open
        snap = snapshot_mark_to_market(
            broker,
            as_of="2024-01-09",
            cache_dir=tmp,
        )
        assert snap["portfolio_equity"] == 50_000.0
        assert any(r["level"] == "strategy" for r in snap["equity_rows"])

        # Stop fills in; position flat at broker
        broker._positions = []
        broker._equity = 49_800.0
        fills = sync_stop_fills(
            broker,
            STRATEGY_S1_EQUITIES,
            since="2024-01-08",
            cache_dir=tmp,
        )
        stop_rows = fills.loc[fills["order_id"].astype(str) == "stop-1"]
        assert len(stop_rows) == 1
        assert str(stop_rows.iloc[0]["fill_type"]) == "stop"
        assert float(stop_rows.iloc[0]["qty"]) == 10.0

        snap2 = snapshot_mark_to_market(
            broker,
            as_of="2024-01-10",
            cache_dir=tmp,
        )
        assert snap2["portfolio_equity"] == 49_800.0


def test_cache_dir_isolation_does_not_touch_default() -> None:
    # Sanity: using an explicit temp cache must not require the default path.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "meta_probe")
        os.makedirs(path, exist_ok=True)
        meta = get_meta(cache_dir=path)
        assert meta["sleeve_weights"][STRATEGY_S1_EQUITIES] == 1.0
        assert os.path.isfile(os.path.join(path, "meta.json"))
