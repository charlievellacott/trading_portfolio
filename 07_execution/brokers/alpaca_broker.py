# imports
import os
import time
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, StopOrderRequest

from data.repo_paths import repo_root

# constants
CREDENTIALS_PATH = os.path.join(repo_root(), "config", "credentials.env")
LIQUIDATE_POLL_INTERVAL_SEC = 2.0
LIQUIDATE_TIMEOUT_SEC = 15 * 60
FILL_POLL_INTERVAL_SEC = 2.0
FILL_TIMEOUT_SEC = 15 * 60
TERMINAL_ORDER_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
}


def _as_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    # pandas.Timestamp and date-like objects
    to_pydt = getattr(value, "to_pydatetime", None)
    if callable(to_pydt):
        return to_pydt()
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def load_alpaca_credentials(path=None):
    creds_path = CREDENTIALS_PATH if path is None else path
    if not os.path.isfile(creds_path):
        raise FileNotFoundError(f"Credentials file not found: {creds_path}")
    api_key = secret_key = None
    with open(creds_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith("ALPACA_API_KEY="):
                api_key = ln.split("=", 1)[1].strip()
            elif ln.startswith("ALPACA_SECRET_KEY="):
                secret_key = ln.split("=", 1)[1].strip()
    if not api_key or not secret_key:
        raise ValueError(
            f"Expected ALPACA_API_KEY=... and ALPACA_SECRET_KEY=... in {creds_path}"
        )
    return api_key, secret_key


# class
class AlpacaBroker:
    # constructor
    def __init__(self, paper: bool = True) -> None:
        # 1. Read key/secret from config/credentials.env
        api_key, secret_key = load_alpaca_credentials()
        # 2. Build TradingClient(paper=paper)
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_account(self):
        return self.client.get_account()

    def get_positions(self):
        return self.client.get_all_positions()

    def get_positions_normalized(self):
        """Plain dict snapshots for live_log mark-to-market."""
        rows = []
        for pos in list(self.get_positions() or []):
            rows.append(
                {
                    "ticker": str(pos.symbol).strip().upper(),
                    "qty": float(pos.qty),
                    "avg_entry": float(getattr(pos, "avg_entry_price", float("nan"))),
                    "market_value": float(getattr(pos, "market_value", float("nan"))),
                    "unrealized_pl": float(
                        getattr(pos, "unrealized_pl", float("nan"))
                    ),
                    "current_price": float(
                        getattr(pos, "current_price", float("nan"))
                    ),
                }
            )
        return rows

    def get_filled_orders(self, after=None, until=None, limit=500):
        """
        Normalized filled orders for stop-fill sync.

        Returns list of dicts:
        ``order_id, ticker, side, qty, price, filled_at, order_type``.
        """
        kwargs = {
            "status": QueryOrderStatus.CLOSED,
            "limit": int(limit),
            "nested": False,
        }
        if after is not None:
            kwargs["after"] = _as_datetime(after)
        if until is not None:
            kwargs["until"] = _as_datetime(until)

        orders = list(self.client.get_orders(GetOrdersRequest(**kwargs)) or [])
        rows = []
        for order in orders:
            filled_qty = float(getattr(order, "filled_qty", 0) or 0)
            if filled_qty <= 0:
                continue
            side = getattr(order, "side", None)
            side_str = side.value if hasattr(side, "value") else str(side)
            otype = getattr(order, "type", None)
            otype_str = otype.value if hasattr(otype, "value") else str(otype)
            filled_at = getattr(order, "filled_at", None) or getattr(
                order, "updated_at", None
            )
            rows.append(
                {
                    "order_id": str(order.id),
                    "ticker": str(order.symbol).strip().upper(),
                    "side": str(side_str).lower(),
                    "qty": filled_qty,
                    "price": float(getattr(order, "filled_avg_price", 0) or 0)
                    or None,
                    "filled_at": filled_at,
                    "order_type": str(otype_str).lower(),
                }
            )
        return rows

    def cancel_open_orders(self):
        return self.client.cancel_orders()


    def cancel_order(self, order_id):
        return self.client.cancel_order_by_id(str(order_id))

    def liquidate_all_positions(self, timeout_sec=LIQUIDATE_TIMEOUT_SEC):
        # Returns {"names": [{"ticker", "qty", "seconds"}, ...], "total_seconds": float}
        t0 = time.monotonic()
        snapshot = []
        for pos in list(self.get_positions() or []):
            snapshot.append(
                {
                    "ticker": str(pos.symbol).strip().upper(),
                    "qty": float(pos.qty),
                }
            )
        if not snapshot:
            return {"names": [], "total_seconds": 0.0}

        pending = {row["ticker"]: row["qty"] for row in snapshot}
        closed = []

        # 1. Cancel resting GTC stops, then full-qty close (incl. fractional)
        self.cancel_open_orders()
        try:
            self.client.close_all_positions(cancel_orders=True)
        except Exception:
            pass

        # 2. Poll until flat; record elapsed when each symbol disappears
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            live = {
                str(pos.symbol).strip().upper(): float(pos.qty)
                for pos in list(self.get_positions() or [])
            }
            now = time.monotonic()
            for ticker in list(pending.keys()):
                if ticker not in live:
                    closed.append(
                        {
                            "ticker": ticker,
                            "qty": pending.pop(ticker),
                            "seconds": now - t0,
                        }
                    )
            if not pending:
                return {
                    "names": closed,
                    "total_seconds": time.monotonic() - t0,
                }
            # 3. Retry leftover names immediately
            for ticker in list(pending.keys()):
                try:
                    self.client.close_position(ticker)
                except Exception:
                    pass
            time.sleep(LIQUIDATE_POLL_INTERVAL_SEC)

        leftover = list(self.get_positions() or [])
        if leftover:
            summary = ", ".join(f"{p.symbol} qty={p.qty}" for p in leftover)
            raise RuntimeError(
                f"liquidate_all_positions: still open after timeout: {summary}"
            )
        now = time.monotonic()
        for ticker, qty in pending.items():
            closed.append({"ticker": ticker, "qty": qty, "seconds": now - t0})
        return {"names": closed, "total_seconds": now - t0}

    def wait_for_order(self, order_id, timeout_sec=FILL_TIMEOUT_SEC):
        deadline = time.monotonic() + float(timeout_sec)
        last = None
        while time.monotonic() < deadline:
            last = self.client.get_order_by_id(str(order_id))
            if last.status in TERMINAL_ORDER_STATUSES:
                return last
            time.sleep(FILL_POLL_INTERVAL_SEC)
        raise TimeoutError(
            f"Order {order_id} did not reach a terminal status within {timeout_sec}s "
            f"(last status: {getattr(last, 'status', None)})"
        )

    def submit_order(
        self,
        ticker,
        size=None,
        direction=None,
        stop_loss=None,
        take_profit=None,
        quantity=None,
    ):
        # DAY market only. Stops go through submit_stop_order after fill.
        if stop_loss is not None or take_profit is not None:
            raise ValueError(
                "submit_order: stop_loss/take_profit not supported; "
                "use submit_stop_order after fill"
            )
        if quantity is not None and size is not None:
            raise ValueError("submit_order: pass quantity or size, not both")
        if quantity is None and size is None:
            raise ValueError("submit_order: quantity or size is required")

        d = str(direction).strip().lower()
        if d in ("buy", "long"):
            side = OrderSide.BUY
        elif d in ("sell", "short"):
            side = OrderSide.SELL
        else:
            raise ValueError(f"unknown direction: {direction!r}")

        kwargs = {
            "symbol": str(ticker).strip().upper(),
            "side": side,
            "time_in_force": TimeInForce.DAY,
        }
        if quantity is not None:
            kwargs["qty"] = quantity
        else:
            kwargs["notional"] = size
        return self.client.submit_order(MarketOrderRequest(**kwargs))

    def submit_stop_order(self, ticker, quantity, direction, stop_price):
        qty = int(quantity)
        if qty < 1:
            raise ValueError(f"submit_stop_order: quantity must be >= 1, got {quantity!r}")
        stop_px = round(float(stop_price), 2)
        if not (stop_px > 0):
            raise ValueError(f"submit_stop_order: stop_price must be positive, got {stop_price!r}")

        d = str(direction).strip().lower()
        if d in ("buy", "long"):
            side = OrderSide.BUY
        elif d in ("sell", "short"):
            side = OrderSide.SELL
        else:
            raise ValueError(f"unknown direction: {direction!r}")

        return self.client.submit_order(
            StopOrderRequest(
                symbol=str(ticker).strip().upper(),
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                stop_price=stop_px,
            )
        )

    def close_uncovered_positions(self, covered_qty_by_ticker):
        # Returns [{"ticker", "qty", "cover", "reason", "closed"}, ...]
        covered = {
            str(k).strip().upper(): float(v)
            for k, v in (covered_qty_by_ticker or {}).items()
        }
        closed = []
        for pos in list(self.get_positions() or []):
            ticker = str(pos.symbol).strip().upper()
            live_qty = abs(float(pos.qty))
            cover = covered.get(ticker, 0.0)
            if live_qty > cover + 1e-8:
                if cover <= 0:
                    reason = f"no GTC stop cover; live_qty={live_qty}"
                else:
                    reason = f"live qty {live_qty} exceeds stop cover {cover}"
                ok = False
                try:
                    self.client.close_position(ticker)
                    ok = True
                except Exception as exc:
                    reason = f"{reason}; close failed: {exc}"
                closed.append(
                    {
                        "ticker": ticker,
                        "qty": live_qty,
                        "cover": cover,
                        "reason": reason,
                        "closed": ok,
                    }
                )
        return closed
