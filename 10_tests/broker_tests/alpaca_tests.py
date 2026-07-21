"""Alpaca broker connectivity and order-lifecycle integration test.

Run:
    python 10_tests/broker_tests/alpaca_tests.py

API keys are read from config/credentials.env (line 1 = key, line 2 = secret).
Uses paper trading by default; pass --live to hit the live endpoint (not recommended).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

CREDENTIALS_PATH = os.path.join(ROOT, "config", "credentials.env")
DEFAULT_SYMBOL = "SPY"
FILL_POLL_INTERVAL_SEC = 2.0
FILL_TIMEOUT_SEC = 120.0
TERMINAL_ORDER_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def load_credentials(path: str = CREDENTIALS_PATH) -> tuple[str, str]:
    """Return (api_key, secret_key) from the first two non-empty lines."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Credentials file not found: {path}")

    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Expected key and secret on lines 1-2 of {path}")

    return lines[0], lines[1]


def make_client(*, paper: bool = True) -> TradingClient:
    api_key, secret_key = load_credentials()
    return TradingClient(api_key, secret_key, paper=paper)


def wait_for_order(
    client: TradingClient,
    order_id: str,
    *,
    timeout_sec: float = FILL_TIMEOUT_SEC,
    poll_interval_sec: float = FILL_POLL_INTERVAL_SEC,
):
    deadline = time.monotonic() + timeout_sec
    last = None
    while time.monotonic() < deadline:
        last = client.get_order_by_id(order_id)
        if last.status in TERMINAL_ORDER_STATUSES:
            return last
        time.sleep(poll_interval_sec)
    raise TimeoutError(
        f"Order {order_id} did not reach a terminal status within {timeout_sec}s "
        f"(last status: {getattr(last, 'status', None)})"
    )


def position_qty(client: TradingClient, symbol: str) -> float:
    try:
        pos = client.get_open_position(symbol)
    except Exception:
        return 0.0
    return float(pos.qty)


def open_orders_for_symbol(client: TradingClient, symbol: str):
    return [o for o in client.get_orders(GetOrdersRequest(status="open")) if o.symbol == symbol]


def run_alpaca_integration(*, paper: bool = True, symbol: str = DEFAULT_SYMBOL) -> list[CheckResult]:
    results: list[CheckResult] = []
    client = make_client(paper=paper)
    mode = "paper" if paper else "LIVE"

    # --- Connectivity: account balances ---
    try:
        account = client.get_account()
        results.append(
            CheckResult(
                "fetch account balances",
                True,
                (
                    f"[{mode}] status={account.status} "
                    f"cash={account.cash} buying_power={account.buying_power} "
                    f"equity={account.equity} portfolio_value={account.portfolio_value}"
                ),
            )
        )
    except Exception as exc:
        results.append(CheckResult("fetch account balances", False, str(exc)))
        return results

    # --- Connectivity: open positions ---
    try:
        positions = client.get_all_positions()
        if positions:
            summary = "; ".join(f"{p.symbol} qty={p.qty} mkt_value={p.market_value}" for p in positions)
        else:
            summary = "(none)"
        results.append(CheckResult("fetch open positions", True, summary))
    except Exception as exc:
        results.append(CheckResult("fetch open positions", False, str(exc)))
        return results

    # --- Place trivial market buy (1 share) ---
    order = None
    try:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        accepted = order.status in {
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PENDING_NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
        }
        results.append(
            CheckResult(
                "order accepted",
                accepted,
                f"id={order.id} status={order.status} symbol={order.symbol} qty={order.qty}",
            )
        )
        if not accepted:
            return results
    except Exception as exc:
        results.append(CheckResult("order accepted", False, str(exc)))
        return results

    # --- Wait for fill / partial fill ---
    try:
        final_order = wait_for_order(client, str(order.id))
        filled_qty = float(final_order.filled_qty or 0)
        fill_ok = final_order.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED} and filled_qty > 0
        results.append(
            CheckResult(
                "fill / partial fill",
                fill_ok,
                (
                    f"status={final_order.status} filled_qty={final_order.filled_qty} "
                    f"filled_avg_price={final_order.filled_avg_price}"
                ),
            )
        )
        if not fill_ok:
            _safe_cancel(client, str(order.id))
            return results
    except Exception as exc:
        results.append(CheckResult("fill / partial fill", False, str(exc)))
        _safe_cancel(client, str(order.id))
        return results

    # --- Position appears ---
    try:
        qty = position_qty(client, symbol)
        pos_ok = qty > 0
        results.append(
            CheckResult(
                "position appears",
                pos_ok,
                f"{symbol} open qty={qty}",
            )
        )
        if not pos_ok:
            return results
    except Exception as exc:
        results.append(CheckResult("position appears", False, str(exc)))
        return results

    # --- Cleanup: close position + cancel stray orders ---
    try:
        client.close_position(symbol)
        close_deadline = time.monotonic() + 60.0
        while time.monotonic() < close_deadline and position_qty(client, symbol) > 0:
            time.sleep(FILL_POLL_INTERVAL_SEC)

        for open_order in open_orders_for_symbol(client, symbol):
            client.cancel_order_by_id(str(open_order.id))

        remaining_qty = position_qty(client, symbol)
        open_for_symbol = open_orders_for_symbol(client, symbol)
        cleanup_ok = remaining_qty == 0 and not open_for_symbol
        results.append(
            CheckResult(
                "order/position cleanup",
                cleanup_ok,
                f"remaining_qty={remaining_qty} open_orders={len(open_for_symbol)}",
            )
        )
    except Exception as exc:
        results.append(CheckResult("order/position cleanup", False, str(exc)))

    return results


def _safe_cancel(client: TradingClient, order_id: str) -> None:
    try:
        client.cancel_order_by_id(order_id)
    except Exception:
        pass


def print_report(results: list[CheckResult]) -> int:
    print("\nAlpaca integration test report")
    print("=" * 40)
    all_passed = True
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {r.name}")
        print(f"       {r.detail}")
        all_passed = all_passed and r.passed
    print("=" * 40)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return 0 if all_passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alpaca connectivity and order lifecycle test")
    parser.add_argument("--live", action="store_true", help="Use live trading endpoint (default: paper)")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help=f"Symbol for test order (default: {DEFAULT_SYMBOL})")
    args = parser.parse_args(argv)

    if args.live:
        print("WARNING: running against LIVE Alpaca endpoint.", file=sys.stderr)

    try:
        results = run_alpaca_integration(paper=not args.live, symbol=args.symbol.upper())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Credential error: {exc}", file=sys.stderr)
        return 2

    return print_report(results)


if __name__ == "__main__":
    raise SystemExit(main())
