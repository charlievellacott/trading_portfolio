# S1 equities Monday paper runner.
# Start 30-60 minutes before the NYSE open (09:30 America/New_York) and leave
# the process running until flatten, new entries, and GTC stops are done.
#
# UK wall clock (most of the year, UK and US DST aligned):
#   NYSE open 14:30 UK → start this script about 13:30-14:00 UK.
# UK wall clock (DST mismatch weeks: US already on EDT / UK still GMT, or
# the late-October reverse):
#   NYSE open 13:30 UK → start this script about 12:30-13:00 UK.
#
# Sizing currently uses full account equity while the live_log sleeve weight for
# s1_equities is 1.0. When a second sleeve is added, size against
# allocated_equity = sleeve_weight * account.equity instead.

# Imports
import os
import time
import datetime
import uuid
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from execution.brokers.alpaca_broker import AlpacaBroker
from performance.live_log import (
    BROKER_ALPACA,
    STRATEGY_S1_EQUITIES,
    log_equity_daily,
    log_fills,
    log_positions_snapshot,
    log_session_fills,
    log_signals,
)
from risk.pct_stop import pct_stop_price
from strategies.s1_equities.s1_strategy import S1Strategy

# Constants
PAPER = True
NY_TZ = ZoneInfo("America/New_York")
NYSE_OPEN_HOUR = 9
NYSE_OPEN_MINUTE = 30
OPEN_POLL_INTERVAL_SEC = 5.0
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "logs")
STRATEGY_ID = STRATEGY_S1_EQUITIES
BROKER_ID = BROKER_ALPACA


# Subroutines
def is_us_equity_trading_day(date: datetime.date) -> bool:
    # NYSE tradable day: weekday Mon-Fri and not a US federal holiday
    if date.weekday() >= 5:  # Sat/Sun
        return False

    # Use pandas to get set of federal holidays for range including today
    cal = USFederalHolidayCalendar()
    # Calendar works on pd.Timestamp, not datetime.date
    holidays = cal.holidays(
        start=pd.Timestamp(date) - pd.Timedelta(days=7),
        end=pd.Timestamp(date) + pd.Timedelta(days=7)
    )
    # NYSE may differ on some holidays (e.g., Good Friday isn't federal)
    # For strict compliance, consider a more specialized NYSE calendar/package
    if pd.Timestamp(date) in holidays:
        return False
    return True


def wait_for_market_open():
    # 1. Poll until local time in America/New_York is >= 09:30 on today
    while True:
        now = datetime.datetime.now(NY_TZ)
        open_dt = now.replace(
            hour=NYSE_OPEN_HOUR,
            minute=NYSE_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        if now >= open_dt:
            return
        time.sleep(OPEN_POLL_INTERVAL_SEC)


def log_line(log_path, msg):
    # 1. Print + append to log file
    line = str(msg)
    print(line)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def place_orders(broker, strategy, weights, log_path, *, run_id: str):
    # 1. Fresh equity after flatten; stop pct from recipe
    # TODO(multi-sleeve): size with allocated_equity = sleeve_weight * equity
    equity = float(broker.get_account().equity)
    pct = float(strategy.STOP_PCT_STAR)

    # 2. Whole-share DAY entries (round down; skip qty < 1)
    pending = []
    t_open_start = time.monotonic()
    for _, row in weights.iterrows():
        ticker = row["ticker"]
        weight = float(row["weight"])
        if weight == 0.0:
            continue
        px = float(row["close"])
        if not (px > 0):
            continue
        direction = "buy" if weight > 0 else "sell"
        qty = int(abs(weight) * equity / px)
        if qty < 1:
            log_line(
                log_path,
                f"SKIP qty<1  {ticker}  weight={weight}  px={px}",
            )
            continue
        order = broker.submit_order(
            ticker=ticker,
            quantity=qty,
            size=None,
            direction=direction,
        )
        pending.append(
            {
                "ticker": ticker,
                "is_long": weight > 0,
                "order_id": order.id,
                "t0": time.monotonic(),
            }
        )

    # 3. Wait for fills; timeout → cancel remainder
    filled = []
    covered = {}
    stop_rows = []
    log_line(log_path, "OPENED POSITIONS")
    for item in pending:
        try:
            final = broker.wait_for_order(item["order_id"])
        except TimeoutError:
            broker.cancel_order(item["order_id"])
            final = broker.wait_for_order(item["order_id"], timeout_sec=60)
        fill_px = float(getattr(final, "filled_avg_price", 0) or 0)
        fill_qty = int(float(getattr(final, "filled_qty", 0) or 0))
        elapsed = time.monotonic() - item["t0"]
        if fill_px <= 0 or fill_qty < 1:
            log_line(
                log_path,
                f"  {item['ticker']}  OPEN FAILED  sec={elapsed:.1f}",
            )
            continue
        log_line(
            log_path,
            f"  {item['ticker']}  qty={fill_qty}  fill={fill_px}  open_sec={elapsed:.1f}",
        )
        filled.append({**item, "fill_px": fill_px, "fill_qty": fill_qty})

    all_open_sec = time.monotonic() - t_open_start
    log_line(log_path, f"  ALL OPENED in {all_open_sec:.1f}s")

    # 4. GTC 20% stops from actual fill; flag failures, keep going
    log_line(log_path, "STOPS")
    for item in filled:
        ticker = item["ticker"]
        stop_px = None
        try:
            stop_px = round(
                pct_stop_price(item["fill_px"], is_long=item["is_long"], pct=pct),
                2,
            )
            stop_side = "sell" if item["is_long"] else "buy"
            stop_order = broker.submit_stop_order(
                ticker=ticker,
                quantity=item["fill_qty"],
                direction=stop_side,
                stop_price=stop_px,
            )
            stop_rows.append(
                {
                    "ticker": ticker,
                    "order_id": stop_order.id,
                    "side": stop_side,
                    "qty": item["fill_qty"],
                    "stop_price": stop_px,
                }
            )
            log_line(
                log_path,
                f"  {ticker}  STOP OK  qty={item['fill_qty']}  "
                f"stop={stop_px}  order_id={stop_order.id}",
            )
        except Exception as exc:
            stop_txt = "n/a" if stop_px is None else str(stop_px)
            log_line(
                log_path,
                f"  {ticker}  STOP FAILED  qty={item['fill_qty']}  "
                f"stop={stop_txt}  err={exc}",
            )
            continue
        covered[str(ticker).upper()] = float(item["fill_qty"])

    # 5. Same-session: close any live qty not fully stop-covered
    uncovered = broker.close_uncovered_positions(covered)
    log_line(log_path, "CLOSED UNCOVERED")
    if not uncovered:
        log_line(log_path, "  none")
    for row in uncovered:
        status = "CLOSED" if row["closed"] else "CLOSE FAILED"
        log_line(
            log_path,
            f"  {row['ticker']}  {status}  qty={row['qty']}  "
            f"cover={row['cover']}  reason={row['reason']}",
        )

    # 6. Structured live_log fills (entries + uncovered). Stop fills sync at EOD.
    log_session_fills(
        strategy_id=STRATEGY_ID,
        broker_id=BROKER_ID,
        run_id=run_id,
        entries=filled,
        uncovered=uncovered,
    )
    # Persist resting stop order ids as audit fills with qty=0? Prefer separate
    # metadata via log_fills only when filled. Store submitted stop ids as
    # zero-price placeholders with fill_type=stop only after a fill at EOD.
    # Keep submitted ids on disk by writing a sentinel row keyed by order_id
    # so sync can associate strategy_id (qty 0 skipped by qty rebuild).
    if stop_rows:
        sentinel = []
        for row in stop_rows:
            sentinel.append(
                {
                    "ts": pd.Timestamp.now(tz="UTC").tz_localize(None),
                    "strategy_id": STRATEGY_ID,
                    "broker_id": BROKER_ID,
                    "ticker": row["ticker"],
                    "side": row["side"],
                    "qty": 0.0,
                    "price": row.get("stop_price"),
                    "order_id": str(row["order_id"]),
                    "fill_type": "stop_submitted",
                    "run_id": run_id,
                }
            )
        log_fills(sentinel)

    return filled, uncovered, stop_rows


def _log_open_book_snapshot(broker, *, run_id: str) -> None:
    positions = broker.get_positions_normalized()
    rows = []
    for pos in positions:
        rows.append(
            {
                "ts": pd.Timestamp.now(tz="UTC").tz_localize(None),
                "strategy_id": STRATEGY_ID,
                "broker_id": BROKER_ID,
                "ticker": pos["ticker"],
                "qty": pos["qty"],
                "avg_entry": pos.get("avg_entry"),
                "market_value": pos.get("market_value"),
                "unrealized_pl": pos.get("unrealized_pl"),
            }
        )
    if rows:
        log_positions_snapshot(rows)

    account = broker.get_account()
    equity = float(account.equity)
    cash = float(getattr(account, "cash", float("nan")))
    day = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    log_equity_daily(
        [
            {
                "date": day,
                "level": "portfolio",
                "strategy_id": "",
                "equity": equity,
                "allocated_equity": equity,
                "cash": cash,
            },
            {
                "date": day,
                "level": "strategy",
                "strategy_id": STRATEGY_ID,
                "equity": equity,
                "allocated_equity": equity,
                "cash": None,
            },
        ]
    )


# Main

def main() -> None:
    # 1. Monday + tradable US equity day
    today = datetime.date.today()
    if today.weekday() != 0:
        raise RuntimeError(f"Today ({today}) is not a Monday. Exiting.")
    if not is_us_equity_trading_day(today):
        raise RuntimeError(
            f"Today ({today}) is not a tradable Monday for US equities (probably a holiday). Exiting."
        )

    broker = AlpacaBroker(paper=PAPER)
    log_path = os.path.join(LOG_DIR, f"s1_paper_{today:%Y%m%d}.txt")
    run_id = str(uuid.uuid4())

    # 2. Predictions + signed weights (pre-open)
    strategy = S1Strategy(start_date=today)
    panel = strategy.generate_features()
    weights = strategy.get_weights(panel=panel)
    log_line(log_path, "PREDICTIONS AND WEIGHTS")
    for _, row in weights.iterrows():
        score = row["score"] if "score" in weights.columns else float("nan")
        log_line(
            log_path,
            f"  {row['ticker']}  score={score}  weight={row['weight']}",
        )
    log_signals(
        STRATEGY_ID,
        weights,
        broker_id=BROKER_ID,
        run_id=run_id,
    )

    # 3. Wait until NYSE 09:30 (runner helper; no-op if already open)
    wait_for_market_open()
    log_line(log_path, "MARKET OPEN")

    # 4. Close previous book; must be flat before new risk
    closed = broker.liquidate_all_positions()
    log_line(log_path, "CLOSED POSITIONS")
    for row in closed["names"]:
        log_line(
            log_path,
            f"  {row['ticker']}  qty={row['qty']}  close_sec={row['seconds']:.1f}",
        )
    log_line(log_path, f"  ALL CLOSED in {closed['total_seconds']:.1f}s")
    log_session_fills(
        strategy_id=STRATEGY_ID,
        broker_id=BROKER_ID,
        run_id=run_id,
        liquidate=closed["names"],
    )

    # 5. Place new book + GTC stops (qty from post-flatten equity)
    place_orders(broker, strategy, weights, log_path, run_id=run_id)

    # 6. Post-session positions + equity for live_log
    _log_open_book_snapshot(broker, run_id=run_id)


if __name__ == "__main__":
    main()
