"""S1 strategy + runner + AlpacaBroker pipeline tester.

Run:
    python 10_tests/s1_pipeline_tester.py
    python 10_tests/s1_pipeline_tester.py --live-strategy
    python 10_tests/s1_pipeline_tester.py --live-strategy --date 2026-08-03
    python 10_tests/s1_pipeline_tester.py --paper
    python 10_tests/s1_pipeline_tester.py --live-strategy --paper

Default is offline FakeBroker + stub weights. ``--live-strategy`` runs real
S1Strategy (slow / network). ``--paper`` places 1 share via AlpacaBroker on
the paper account (does not flatten the whole book).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "s1_logs")
_TEST_CACHE_DIR = os.path.join(LOG_DIR, "_cache")
os.environ.setdefault("S1_CACHE_DIR", os.path.abspath(_TEST_CACHE_DIR))

from execution.brokers.alpaca_broker import AlpacaBroker
from execution.s1_equities.s1_paper_runner import (
    is_us_equity_trading_day,
    place_orders,
)
from risk.pct_stop import pct_stop_price
from strategies.s1_equities.s1_strategy import S1Strategy

DEFAULT_PAPER_SYMBOL = "SPY"
STUB_EQUITY = 100_000.0
STOP_PCT = 20.0
PAPER_FILL_TIMEOUT_SEC = 90.0


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    skipped: bool = False


class FakeBroker:
    """Duck-typed stand-in for AlpacaBroker used by place_orders."""

    def __init__(
        self,
        equity=STUB_EQUITY,
        fill_prices=None,
        fail_stop_tickers=None,
    ) -> None:
        self.equity = float(equity)
        self.fill_prices = {
            str(k).strip().upper(): float(v) for k, v in (fill_prices or {}).items()
        }
        self.fail_stop_tickers = {
            str(t).strip().upper() for t in (fail_stop_tickers or set())
        }
        self.positions: dict[str, float] = {}
        self.orders: dict[str, SimpleNamespace] = {}
        self.stops: dict[str, SimpleNamespace] = {}
        self.canceled_ids: set[str] = set()
        self._next_id = 1

    def get_account(self):
        return SimpleNamespace(equity=self.equity)

    def get_positions(self):
        return [
            SimpleNamespace(symbol=ticker, qty=qty)
            for ticker, qty in self.positions.items()
            if abs(qty) > 1e-12
        ]

    def cancel_order(self, order_id):
        oid = str(order_id)
        self.canceled_ids.add(oid)
        order = self.orders.get(oid)
        if order is not None:
            order.status = "canceled"
        return order

    def wait_for_order(self, order_id, timeout_sec=None):
        oid = str(order_id)
        order = self.orders.get(oid)
        if order is None:
            raise KeyError(f"unknown order_id: {oid}")
        return order

    def submit_order(
        self,
        ticker,
        size=None,
        direction=None,
        stop_loss=None,
        take_profit=None,
        quantity=None,
    ):
        if stop_loss is not None or take_profit is not None:
            raise ValueError("submit_order: stop_loss/take_profit not supported")
        if quantity is not None and size is not None:
            raise ValueError("submit_order: pass quantity or size, not both")
        if quantity is None and size is None:
            raise ValueError("submit_order: quantity or size is required")

        tkr = str(ticker).strip().upper()
        d = str(direction).strip().lower()
        if d not in ("buy", "long", "sell", "short"):
            raise ValueError(f"unknown direction: {direction!r}")
        qty = int(quantity) if quantity is not None else 0
        if qty < 1:
            raise ValueError(f"submit_order: quantity must be >= 1, got {quantity!r}")
        px = self.fill_prices.get(tkr)
        if px is None or not (px > 0):
            raise ValueError(f"FakeBroker: no fill price for {tkr}")

        signed = float(qty) if d in ("buy", "long") else -float(qty)
        self.positions[tkr] = self.positions.get(tkr, 0.0) + signed

        oid = str(self._next_id)
        self._next_id += 1
        order = SimpleNamespace(
            id=oid,
            ticker=tkr,
            quantity=qty,
            direction=d,
            status="filled",
            filled_avg_price=float(px),
            filled_qty=float(qty),
        )
        self.orders[oid] = order
        return order

    def submit_stop_order(self, ticker, quantity, direction, stop_price):
        tkr = str(ticker).strip().upper()
        if tkr in self.fail_stop_tickers:
            raise RuntimeError(f"injected stop failure for {tkr}")
        qty = int(quantity)
        if qty < 1:
            raise ValueError(f"submit_stop_order: quantity must be >= 1, got {quantity!r}")
        stop_px = round(float(stop_price), 2)
        if not (stop_px > 0):
            raise ValueError(
                f"submit_stop_order: stop_price must be positive, got {stop_price!r}"
            )
        d = str(direction).strip().lower()
        if d not in ("buy", "long", "sell", "short"):
            raise ValueError(f"unknown direction: {direction!r}")
        oid = str(self._next_id)
        self._next_id += 1
        stop = SimpleNamespace(
            id=oid,
            ticker=tkr,
            quantity=qty,
            direction=d,
            stop_price=stop_px,
            status="open",
        )
        self.stops[tkr] = stop
        self.orders[oid] = stop
        return stop

    def liquidate_all_positions(self, timeout_sec=None):
        t0 = time.monotonic()
        snapshot = [
            {"ticker": ticker, "qty": float(qty)}
            for ticker, qty in list(self.positions.items())
            if abs(qty) > 1e-12
        ]
        if not snapshot:
            return {"names": [], "total_seconds": 0.0}
        self.stops.clear()
        self.positions.clear()
        elapsed = time.monotonic() - t0
        return {
            "names": [
                {"ticker": row["ticker"], "qty": row["qty"], "seconds": elapsed}
                for row in snapshot
            ],
            "total_seconds": elapsed,
        }

    def close_uncovered_positions(self, covered_qty_by_ticker):
        covered = {
            str(k).strip().upper(): float(v)
            for k, v in (covered_qty_by_ticker or {}).items()
        }
        closed = []
        for ticker, qty in list(self.positions.items()):
            live_qty = abs(float(qty))
            cover = covered.get(ticker, 0.0)
            if live_qty > cover + 1e-8:
                if cover <= 0:
                    reason = f"no GTC stop cover; live_qty={live_qty}"
                else:
                    reason = f"live qty {live_qty} exceeds stop cover {cover}"
                del self.positions[ticker]
                closed.append(
                    {
                        "ticker": ticker,
                        "qty": live_qty,
                        "cover": cover,
                        "reason": reason,
                        "closed": True,
                    }
                )
        return closed


def stub_weights() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "LONG", "weight": 0.20, "close": 50.0, "score": 1.0},
            {"ticker": "SHRT", "weight": -0.15, "close": 25.0, "score": -1.0},
            {"ticker": "TINY", "weight": 0.001, "close": 200.0, "score": 0.5},
            {"ticker": "ZERO", "weight": 0.0, "close": 10.0, "score": 0.0},
            {"ticker": "BAD", "weight": 0.10, "close": 40.0, "score": 0.8},
        ]
    )


def expected_qtys(weights: pd.DataFrame, equity: float) -> dict[str, dict]:
    out = {}
    for _, row in weights.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        weight = float(row["weight"])
        px = float(row["close"])
        if weight == 0.0 or not (px > 0):
            continue
        qty = int(abs(weight) * equity / px)
        if qty < 1:
            continue
        out[ticker] = {"qty": qty, "is_long": weight > 0, "px": px}
    return out


def most_recent_monday(today: datetime.date | None = None) -> datetime.date:
    d = today or datetime.date.today()
    return d - datetime.timedelta(days=d.weekday())


def dump_weights(weights: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in weights.iterrows():
        rec = {
            "ticker": str(row["ticker"]).strip().upper(),
            "weight": float(row["weight"]),
            "close": float(row["close"]) if "close" in weights.columns else None,
        }
        if "score" in weights.columns and pd.notna(row["score"]):
            rec["score"] = float(row["score"])
        rows.append(rec)
    return rows


def dump_fake_broker(broker: FakeBroker) -> dict:
    return {
        "positions": {k: float(v) for k, v in broker.positions.items()},
        "stops": {
            tkr: {
                "qty": int(stop.quantity),
                "direction": stop.direction,
                "stop_price": float(stop.stop_price),
            }
            for tkr, stop in broker.stops.items()
        },
        "fills": [
            {
                "id": order.id,
                "ticker": getattr(order, "ticker", None),
                "filled_qty": float(getattr(order, "filled_qty", 0) or 0),
                "filled_avg_price": float(getattr(order, "filled_avg_price", 0) or 0),
                "status": getattr(order, "status", None),
            }
            for order in broker.orders.values()
            if getattr(order, "filled_avg_price", None) is not None
        ],
    }


def _cleanup_paper_symbol(broker: AlpacaBroker, symbol: str, extra_order_ids=()) -> None:
    for oid in extra_order_ids:
        try:
            broker.cancel_order(oid)
        except Exception:
            pass
    try:
        from alpaca.trading.requests import GetOrdersRequest

        opens = broker.client.get_orders(GetOrdersRequest(status="open"))
        for order in opens or []:
            if str(getattr(order, "symbol", "")).strip().upper() == symbol:
                try:
                    broker.cancel_order(order.id)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        broker.client.close_position(symbol)
    except Exception:
        pass


def check_calendar() -> list[CheckResult]:
    results = []
    weekday = datetime.date(2026, 8, 5)  # Wednesday
    weekend = datetime.date(2026, 8, 8)  # Saturday
    holiday = datetime.date(2026, 1, 1)  # New Year's Day

    wd_ok = is_us_equity_trading_day(weekday)
    results.append(
        CheckResult(
            "calendar weekday",
            wd_ok,
            f"{weekday.isoformat()} tradable={wd_ok} (expected True)",
        )
    )
    we_ok = not is_us_equity_trading_day(weekend)
    results.append(
        CheckResult(
            "calendar weekend",
            we_ok,
            f"{weekend.isoformat()} tradable={is_us_equity_trading_day(weekend)} (expected False)",
        )
    )
    hol_ok = not is_us_equity_trading_day(holiday)
    results.append(
        CheckResult(
            "calendar holiday",
            hol_ok,
            f"{holiday.isoformat()} tradable={is_us_equity_trading_day(holiday)} (expected False)",
        )
    )
    return results


def check_stub_place_orders(log_path: str) -> tuple[list[CheckResult], FakeBroker, pd.DataFrame]:
    results = []
    weights = stub_weights()
    fill_prices = {
        str(row["ticker"]).upper(): float(row["close"]) for _, row in weights.iterrows()
    }
    broker = FakeBroker(
        equity=STUB_EQUITY,
        fill_prices=fill_prices,
        fail_stop_tickers={"BAD"},
    )
    strategy = SimpleNamespace(STOP_PCT_STAR=STOP_PCT)
    place_orders(broker, strategy, weights, log_path)

    expected = expected_qtys(weights, STUB_EQUITY)
    log_txt = ""
    if os.path.isfile(log_path):
        with open(log_path, encoding="utf-8") as f:
            log_txt = f.read()

    fill_ok = True
    fill_bits = []
    for ticker, spec in expected.items():
        if ticker == "BAD":
            continue
        pos_qty = abs(float(broker.positions.get(ticker, 0.0)))
        signed = broker.positions.get(ticker, 0.0)
        want_long = spec["is_long"]
        sign_ok = (signed > 0) if want_long else (signed < 0)
        qty_ok = int(pos_qty) == spec["qty"]
        ok = qty_ok and sign_ok
        fill_ok = fill_ok and ok
        fill_bits.append(
            f"{ticker} qty={int(pos_qty)}/{spec['qty']} "
            f"{'long' if signed > 0 else 'short'}"
        )
    fill_tickers = {
        str(getattr(order, "ticker", "")).strip().upper()
        for order in broker.orders.values()
        if getattr(order, "filled_avg_price", None) is not None
    }
    tiny_skipped = "TINY" not in fill_tickers and "TINY" not in broker.positions
    zero_skipped = "ZERO" not in fill_tickers and "ZERO" not in broker.positions
    skip_log = "SKIP qty<1" in log_txt and "TINY" in log_txt
    fill_ok = fill_ok and tiny_skipped and zero_skipped and skip_log
    results.append(
        CheckResult(
            "stub place_orders fills",
            fill_ok,
            "; ".join(fill_bits)
            + f"; TINY skipped={tiny_skipped}; ZERO skipped={zero_skipped}; "
            f"skip_log={skip_log}",
        )
    )

    stop_ok = True
    stop_bits = []
    for ticker, spec in expected.items():
        if ticker == "BAD":
            continue
        stop = broker.stops.get(ticker)
        if stop is None:
            stop_ok = False
            stop_bits.append(f"{ticker} missing stop")
            continue
        want_px = round(
            pct_stop_price(spec["px"], is_long=spec["is_long"], pct=STOP_PCT),
            2,
        )
        want_side = "sell" if spec["is_long"] else "buy"
        px_ok = float(stop.stop_price) == want_px
        side_ok = str(stop.direction).lower() == want_side
        qty_ok = int(stop.quantity) == spec["qty"]
        ok = px_ok and side_ok and qty_ok
        stop_ok = stop_ok and ok
        stop_bits.append(
            f"{ticker} stop={stop.stop_price} (want {want_px}) "
            f"side={stop.direction} (want {want_side})"
        )
    results.append(
        CheckResult("GTC stop px / side", stop_ok, "; ".join(stop_bits))
    )

    bad_closed = "BAD" not in broker.positions
    bad_no_stop = "BAD" not in broker.stops
    uncover_log = "STOP FAILED" in log_txt and "BAD" in log_txt and "CLOSED UNCOVERED" in log_txt
    uncover_ok = bad_closed and bad_no_stop and uncover_log
    results.append(
        CheckResult(
            "uncover-close on stop fail",
            uncover_ok,
            f"BAD closed={bad_closed} no_stop={bad_no_stop} log={uncover_log}",
        )
    )
    return results, broker, weights


def check_fake_liquidate() -> CheckResult:
    broker = FakeBroker(equity=STUB_EQUITY, fill_prices={"AAA": 10.0, "BBB": 20.0})
    broker.positions = {"AAA": 5.0, "BBB": -3.0}
    broker.stops = {
        "AAA": SimpleNamespace(id="s1", ticker="AAA", quantity=5, direction="sell", stop_price=8.0)
    }
    closed = broker.liquidate_all_positions()
    names = {row["ticker"] for row in closed["names"]}
    ok = (
        names == {"AAA", "BBB"}
        and not broker.positions
        and not broker.stops
        and len(closed["names"]) == 2
    )
    return CheckResult(
        "fake liquidate_all_positions",
        ok,
        f"closed={sorted(names)} leftover_pos={list(broker.positions)} "
        f"leftover_stops={list(broker.stops)}",
    )


def check_live_strategy(
    decision_date: datetime.date,
    log_path: str,
) -> tuple[list[CheckResult], pd.DataFrame | None, FakeBroker | None]:
    results = []
    print(f"Running live S1Strategy for {decision_date.isoformat()} (fetch + features + weights)...")
    try:
        strategy = S1Strategy(start_date=decision_date.isoformat())
        panel = strategy.generate_features()
        weights = strategy.get_weights(panel=panel)
    except Exception as exc:
        results.append(CheckResult("live strategy run", False, str(exc)))
        return results, None, None

    feature_cols = list(strategy.feature_cols or [])
    missing_cols = [c for c in feature_cols if c not in panel.columns]
    results.append(
        CheckResult(
            "live feature_cols present",
            not missing_cols and bool(feature_cols),
            f"n_cols={len(feature_cols)} missing={missing_cols[:8]}",
        )
    )

    dt = pd.Timestamp(decision_date).normalize()
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    day = panel.loc[panel["date"] == dt]
    got = set(day["ticker"].astype(str).str.strip().str.upper())
    want = [str(t).strip().upper() for t in strategy.tickers]
    missing_tickers = [t for t in want if t not in got]
    results.append(
        CheckResult(
            "live decision-date rows",
            not missing_tickers and len(day) > 0,
            f"rows={len(day)} missing={missing_tickers[:8]}",
        )
    )

    nz = weights.loc[weights["weight"].astype(float) != 0.0].copy()
    n_nz = int(len(nz))
    gross = float(nz["weight"].abs().sum()) if n_nz else 0.0
    n_ok = n_nz <= int(strategy.N_STAR)
    g_ok = gross <= float(strategy.MAX_GROSS) + 1e-9
    results.append(
        CheckResult(
            "live weight count / gross",
            n_ok and g_ok,
            f"n_nonzero={n_nz} N_STAR={strategy.N_STAR} gross={gross:.4f} "
            f"MAX_GROSS={strategy.MAX_GROSS}",
        )
    )

    sign_ok = True
    sign_detail = "no long/short pair"
    if "score" in nz.columns and not nz.empty:
        longs = nz.loc[nz["weight"].astype(float) > 0]
        shorts = nz.loc[nz["weight"].astype(float) < 0]
        if not longs.empty and not shorts.empty:
            min_long = float(longs["score"].min())
            max_short = float(shorts["score"].max())
            sign_ok = min_long >= max_short
            sign_detail = f"min_long_score={min_long:.6f} max_short_score={max_short:.6f}"
        elif not longs.empty:
            sign_detail = f"longs_only n={len(longs)}"
        elif not shorts.empty:
            sign_detail = f"shorts_only n={len(shorts)}"
    results.append(CheckResult("live long/short vs score", sign_ok, sign_detail))

    fill_prices = {}
    for _, row in weights.iterrows():
        px = float(row["close"]) if "close" in weights.columns else 0.0
        if px > 0:
            fill_prices[str(row["ticker"]).strip().upper()] = px
    broker = FakeBroker(equity=STUB_EQUITY, fill_prices=fill_prices)
    try:
        place_orders(broker, strategy, weights, log_path)
    except Exception as exc:
        results.append(CheckResult("live FakeBroker place_orders", False, str(exc)))
        return results, weights, broker

    exp = expected_qtys(weights, STUB_EQUITY)
    placed_ok = True
    bits = []
    for ticker, spec in exp.items():
        pos = abs(float(broker.positions.get(ticker, 0.0)))
        stop = broker.stops.get(ticker)
        ok = int(pos) == spec["qty"] and stop is not None
        if stop is not None:
            want_px = round(
                pct_stop_price(spec["px"], is_long=spec["is_long"], pct=float(strategy.STOP_PCT_STAR)),
                2,
            )
            want_side = "sell" if spec["is_long"] else "buy"
            ok = ok and float(stop.stop_price) == want_px and str(stop.direction).lower() == want_side
        placed_ok = placed_ok and ok
        bits.append(f"{ticker} qty={int(pos)} stop={'yes' if stop else 'no'}")
    results.append(
        CheckResult(
            "live FakeBroker place_orders",
            placed_ok and bool(exp),
            f"n_expected={len(exp)}; " + "; ".join(bits[:8]),
        )
    )
    return results, weights, broker


def check_paper(symbol: str) -> tuple[list[CheckResult], dict]:
    results = []
    dump = {"symbol": symbol}
    extra_ids = []
    broker = None
    try:
        broker = AlpacaBroker(paper=True)
        account = broker.get_account()
        results.append(
            CheckResult(
                "alpaca paper account",
                True,
                f"status={getattr(account, 'status', None)} equity={account.equity}",
            )
        )
        dump["equity"] = str(getattr(account, "equity", None))
    except Exception as exc:
        results.append(CheckResult("alpaca paper account", False, str(exc)))
        return results, dump

    try:
        positions = broker.get_positions()
        n_pos = len(list(positions or []))
        results.append(CheckResult("alpaca paper positions", True, f"n_open={n_pos}"))
    except Exception as exc:
        results.append(CheckResult("alpaca paper positions", False, str(exc)))
        return results, dump

    market_open = None
    try:
        clock = broker.client.get_clock()
        market_open = bool(clock.is_open)
        dump["market_open"] = market_open
    except Exception as exc:
        dump["clock_error"] = str(exc)

    if market_open is False:
        results.append(
            CheckResult(
                "alpaca paper fill",
                False,
                f"US equity market is closed; cannot fill {symbol}. Retry while NYSE is open.",
            )
        )
        return results, dump

    order = None
    try:
        order = broker.submit_order(
            ticker=symbol,
            quantity=1,
            size=None,
            direction="buy",
        )
        extra_ids.append(order.id)
        results.append(
            CheckResult(
                "alpaca paper order accepted",
                True,
                f"id={order.id} status={getattr(order, 'status', None)} symbol={symbol}",
            )
        )
    except Exception as exc:
        results.append(CheckResult("alpaca paper order accepted", False, str(exc)))
        _cleanup_paper_symbol(broker, symbol, extra_ids)
        return results, dump

    try:
        final = broker.wait_for_order(order.id, timeout_sec=PAPER_FILL_TIMEOUT_SEC)
        fill_px = float(getattr(final, "filled_avg_price", 0) or 0)
        fill_qty = int(float(getattr(final, "filled_qty", 0) or 0))
        fill_ok = fill_px > 0 and fill_qty >= 1
        dump["fill_px"] = fill_px
        dump["fill_qty"] = fill_qty
        results.append(
            CheckResult(
                "alpaca paper fill",
                fill_ok,
                f"status={getattr(final, 'status', None)} qty={fill_qty} px={fill_px}",
            )
        )
        if not fill_ok:
            _cleanup_paper_symbol(broker, symbol, extra_ids)
            return results, dump
    except Exception as exc:
        msg = str(exc)
        if market_open is False or "closed" in msg.lower():
            msg = f"US equity market is closed or order timed out: {exc}"
        results.append(CheckResult("alpaca paper fill", False, msg))
        _cleanup_paper_symbol(broker, symbol, extra_ids)
        return results, dump

    stop_order = None
    try:
        stop_px = round(pct_stop_price(fill_px, is_long=True, pct=STOP_PCT), 2)
        stop_order = broker.submit_stop_order(
            ticker=symbol,
            quantity=fill_qty,
            direction="sell",
            stop_price=stop_px,
        )
        extra_ids.append(stop_order.id)
        dump["stop_px"] = stop_px
        dump["stop_id"] = str(stop_order.id)
        results.append(
            CheckResult(
                "alpaca paper GTC stop",
                True,
                f"id={stop_order.id} stop={stop_px} status={getattr(stop_order, 'status', None)}",
            )
        )
    except Exception as exc:
        results.append(CheckResult("alpaca paper GTC stop", False, str(exc)))
        _cleanup_paper_symbol(broker, symbol, extra_ids)
        return results, dump

    try:
        uncovered = broker.close_uncovered_positions({symbol: float(fill_qty)})
        dump["uncovered"] = uncovered
        results.append(
            CheckResult(
                "alpaca paper uncover no-op",
                not uncovered,
                f"uncovered={uncovered}",
            )
        )
    except Exception as exc:
        results.append(CheckResult("alpaca paper uncover no-op", False, str(exc)))

    _cleanup_paper_symbol(broker, symbol, extra_ids)
    leftover_qty = 0.0
    try:
        leftover_qty = abs(
            float(
                next(
                    (
                        p.qty
                        for p in (broker.get_positions() or [])
                        if str(p.symbol).strip().upper() == symbol
                    ),
                    0.0,
                )
            )
        )
    except Exception:
        leftover_qty = -1.0
    results.append(
        CheckResult(
            "alpaca paper cleanup",
            leftover_qty == 0.0,
            f"{symbol} leftover_qty={leftover_qty}",
        )
    )
    return results, dump


def print_report(results: list[CheckResult]) -> int:
    print("\nS1 pipeline tester")
    print("=" * 40)
    all_passed = True
    for r in results:
        if r.skipped:
            mark = "SKIP"
        elif r.passed:
            mark = "PASS"
        else:
            mark = "FAIL"
            all_passed = False
        print(f"[{mark}] {r.name}")
        print(f"       {r.detail}")
    print("=" * 40)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return 0 if all_passed else 1


def paper_symbol_from_weights(weights: pd.DataFrame | None, explicit: str | None) -> str:
    if explicit:
        return str(explicit).strip().upper()
    if weights is None or weights.empty or "weight" not in weights.columns:
        return DEFAULT_PAPER_SYMBOL
    nz = weights.loc[weights["weight"].astype(float) != 0.0]
    if nz.empty:
        return DEFAULT_PAPER_SYMBOL
    idx = nz["weight"].astype(float).abs().idxmax()
    return str(nz.loc[idx, "ticker"]).strip().upper()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S1 strategy / runner / Alpaca pipeline tester")
    parser.add_argument(
        "--live-strategy",
        action="store_true",
        help="Run real S1Strategy.generate_features + get_weights (slow / network)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Place 1 share via AlpacaBroker paper (does not flatten the book)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Decision date YYYY-MM-DD for --live-strategy (default: most recent Monday)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help=f"Paper symbol (default: {DEFAULT_PAPER_SYMBOL}, or top |weight| with --live-strategy)",
    )
    args = parser.parse_args(argv)

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(LOG_DIR, f"s1_tester_{stamp}.txt")
    json_path = os.path.join(LOG_DIR, f"s1_tester_{stamp}.json")

    if args.date:
        decision_date = pd.Timestamp(args.date).date()
    else:
        decision_date = most_recent_monday()

    dump = {
        "mode": {
            "live_strategy": bool(args.live_strategy),
            "paper": bool(args.paper),
            "date": decision_date.isoformat(),
            "s1_cache_dir": os.environ.get("S1_CACHE_DIR"),
        },
        "checks": [],
        "stub": None,
        "live": None,
        "paper": None,
    }

    results: list[CheckResult] = []
    results.append(
        CheckResult(
            "runner importable",
            True,
            "execution.s1_equities.s1_paper_runner.place_orders",
        )
    )
    results.extend(check_calendar())

    stub_results, stub_broker, stub_w = check_stub_place_orders(log_path)
    results.extend(stub_results)
    dump["stub"] = {
        "weights": dump_weights(stub_w),
        "broker": dump_fake_broker(stub_broker),
    }
    results.append(check_fake_liquidate())

    live_weights = None
    if args.live_strategy:
        live_results, live_weights, live_broker = check_live_strategy(decision_date, log_path)
        results.extend(live_results)
        dump["live"] = {
            "weights": dump_weights(live_weights) if live_weights is not None else None,
            "broker": dump_fake_broker(live_broker) if live_broker is not None else None,
        }
    else:
        results.append(
            CheckResult(
                "live strategy",
                True,
                "pass --live-strategy",
                skipped=True,
            )
        )

    if args.paper:
        symbol = paper_symbol_from_weights(live_weights, args.symbol)
        paper_results, paper_dump = check_paper(symbol)
        results.extend(paper_results)
        dump["paper"] = paper_dump
    else:
        results.append(
            CheckResult(
                "alpaca paper",
                True,
                "pass --paper",
                skipped=True,
            )
        )

    dump["checks"] = [
        {
            "name": r.name,
            "passed": r.passed,
            "skipped": r.skipped,
            "detail": r.detail,
        }
        for r in results
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, default=str)

    print(f"\nLog:  {log_path}")
    print(f"JSON: {json_path}")
    return print_report(results)


if __name__ == "__main__":
    raise SystemExit(main())
