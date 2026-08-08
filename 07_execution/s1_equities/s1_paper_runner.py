# S1 equities Monday paper runner.
# Start 30-60 minutes before the NYSE open (09:30 America/New_York) and leave
# the process running until flatten, new entries, and GTC stops are done.
#
# UK wall clock (most of the year, UK and US DST aligned):
#   NYSE open 14:30 UK → start this script about 13:30-14:00 UK.
# UK wall clock (DST mismatch weeks: US already on EDT / UK still GMT, or
# the late-October reverse):
#   NYSE open 13:30 UK → start this script about 12:30-13:00 UK.

# Imports
import os
import time
import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from execution.brokers.alpaca_broker import AlpacaBroker
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


def place_orders(broker, strategy, weights, log_path):
    # 1. Fresh equity after flatten; stop pct from recipe
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
            broker.submit_stop_order(
                ticker=ticker,
                quantity=item["fill_qty"],
                direction=stop_side,
                stop_price=stop_px,
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

    # 5. Place new book + GTC stops (qty from post-flatten equity)
    place_orders(broker, strategy, weights, log_path)


if __name__ == "__main__":
    main()
