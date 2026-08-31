# S2 cointegration paper runner (Alpaca). Does not import or run S1.
#
# Preferred: start 30-60 minutes before the NYSE open (09:30 America/New_York)
# on the fill morning. Features use the last completed close t. Orders are
# submitted just before the open (09:28 ET DAY market) so they are eligible
# at the opening auction. Late start is OK while RTH is still open. Do not
# run after the US close.
#
# Dedicated S2 paper account only. Never flatten the whole account (a
# hold-to-mean book must not copy the S1 Monday reset, and must not touch S1).
#
# Dry-run: python -m execution.s2_coint.s2_paper_runner --dry-run

from __future__ import annotations

import argparse
import datetime
import os
import time
import uuid
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from execution.brokers.alpaca_broker import (
    AlpacaBroker,
    S2_ALPACA_API_KEY,
    S2_ALPACA_SECRET_KEY,
)
from data.repo_paths import repo_root
from performance.live_log import (
    BROKER_ALPACA,
    STRATEGY_S2_COINT,
    log_equity_daily,
    log_positions_snapshot,
    log_session_fills,
    log_signals,
)
from strategies.s2_coint.s2_strategy import S2Strategy

# Constants
PAPER = True
NY_TZ = ZoneInfo("America/New_York")
NYSE_OPEN_HOUR = 9
NYSE_OPEN_MINUTE = 30
SUBMIT_HOUR = 9
SUBMIT_MINUTE = 28
NYSE_CLOSE_HOUR = 16
NYSE_CLOSE_MINUTE = 0
OPEN_POLL_INTERVAL_SEC = 5.0
DRY_RUN_EQUITY_DEFAULT = 100_000.0
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "logs")
LIVE_LOG_DIR = os.path.join(repo_root(), "09_performance", "cache", "live_s2")
STRATEGY_ID = STRATEGY_S2_COINT
BROKER_ID = BROKER_ALPACA


def is_us_equity_trading_day(date: datetime.date) -> bool:
    if date.weekday() >= 5:
        return False
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(
        start=pd.Timestamp(date) - pd.Timedelta(days=7),
        end=pd.Timestamp(date) + pd.Timedelta(days=7),
    )
    if pd.Timestamp(date) in holidays:
        return False
    return True


def _nyse_session_bounds(now: datetime.datetime):
    open_dt = now.replace(
        hour=NYSE_OPEN_HOUR,
        minute=NYSE_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )
    close_dt = now.replace(
        hour=NYSE_CLOSE_HOUR,
        minute=NYSE_CLOSE_MINUTE,
        second=0,
        microsecond=0,
    )
    submit_dt = now.replace(
        hour=SUBMIT_HOUR,
        minute=SUBMIT_MINUTE,
        second=0,
        microsecond=0,
    )
    return open_dt, close_dt, submit_dt


def ensure_nyse_preopen_submit():
    """Wait until 09:28 ET; allow RTH; refuse after the close."""
    while True:
        now = datetime.datetime.now(NY_TZ)
        _open_dt, close_dt, submit_dt = _nyse_session_bounds(now)
        if now < submit_dt:
            time.sleep(OPEN_POLL_INTERVAL_SEC)
            continue
        if now >= close_dt:
            raise RuntimeError(
                f"NYSE regular session is closed ({now.isoformat()}). "
                "Refusing to submit DAY entries that would queue into the next session."
            )
        return


def log_line(log_path, msg):
    line = str(msg)
    print(line)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _signed_qty(weight: float, equity: float, px: float) -> int:
    if not (px > 0) or weight == 0.0 or not (equity > 0):
        return 0
    qty = int(abs(float(weight)) * float(equity) / float(px))
    if qty < 1:
        return 0
    return qty if weight > 0 else -qty


def _trunc_int(qty: float) -> int:
    q = float(qty)
    if q >= 0:
        return int(q)
    return -int(abs(q))


def planned_orders(
    weights: pd.DataFrame,
    *,
    equity: float,
    current: dict[str, float],
    universe: list[str],
) -> list[dict]:
    """Target vs live qty deltas. Weight 0 → flatten that name."""
    target: dict[str, int] = {str(t).upper(): 0 for t in universe}
    px_by_ticker: dict[str, float] = {}
    for _, row in weights.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        w = float(row["weight"])
        px = float(row["close"]) if pd.notna(row.get("close", float("nan"))) else float("nan")
        px_by_ticker[ticker] = px
        target[ticker] = _signed_qty(w, equity, px)
        if w != 0.0 and target[ticker] == 0:
            target[ticker] = 0

    names = set(target) | {str(k).strip().upper() for k in current}
    plans = []
    for ticker in sorted(names):
        tgt = int(target.get(ticker, 0))
        cur = _trunc_int(current.get(ticker, 0.0))
        delta = tgt - cur
        if delta == 0:
            continue
        direction = "buy" if delta > 0 else "sell"
        plans.append(
            {
                "ticker": ticker,
                "current_qty": cur,
                "target_qty": tgt,
                "delta": delta,
                "direction": direction,
                "quantity": abs(int(delta)),
                "px": px_by_ticker.get(ticker, float("nan")),
                "is_long": direction == "buy",
            }
        )
    return plans


def place_orders(
    broker,
    plans: list[dict],
    log_path: str,
    *,
    run_id: str,
    dry_run: bool = False,
    live_log_dir: str | None = None,
) -> list[dict]:
    if dry_run:
        log_line(log_path, "DRY-RUN ORDERS (not submitted)")
        if not plans:
            log_line(log_path, "  none")
        for p in plans:
            log_line(
                log_path,
                f"  {p['direction'].upper()} {p['quantity']} {p['ticker']}  "
                f"current={p['current_qty']} target={p['target_qty']}",
            )
        return []

    pending = []
    t0 = time.monotonic()
    sells = [p for p in plans if p["direction"] == "sell"]
    buys = [p for p in plans if p["direction"] == "buy"]
    for p in sells + buys:
        order = broker.submit_order(
            ticker=p["ticker"],
            quantity=int(p["quantity"]),
            size=None,
            direction=p["direction"],
        )
        pending.append({**p, "order_id": order.id, "t0": time.monotonic()})

    filled = []
    log_line(log_path, "FILLS")
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
                f"  {item['ticker']}  FILL FAILED  sec={elapsed:.1f}",
            )
            continue
        log_line(
            log_path,
            f"  {item['ticker']}  {item['direction']} qty={fill_qty}  "
            f"fill={fill_px}  sec={elapsed:.1f}",
        )
        filled.append({**item, "fill_px": fill_px, "fill_qty": fill_qty})
    log_line(log_path, f"  ALL FILLS in {time.monotonic() - t0:.1f}s")
    log_session_fills(
        strategy_id=STRATEGY_ID,
        broker_id=BROKER_ID,
        run_id=run_id,
        entries=filled,
        cache_dir=live_log_dir,
    )
    return filled


def _log_open_book_snapshot(broker, *, run_id: str, live_log_dir: str) -> None:
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
        log_positions_snapshot(rows, cache_dir=live_log_dir)
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
        ],
        cache_dir=live_log_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S2 cointegration Alpaca paper runner (not S1)."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate signals and print orders; do not connect or submit.",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Fill date YYYY-MM-DD (default: today in America/New_York).",
    )
    p.add_argument(
        "--skip-wait",
        action="store_true",
        help="Do not wait for the 09:28 ET submit window.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.date:
        today = pd.Timestamp(args.date).date()
    else:
        today = datetime.datetime.now(NY_TZ).date()
    if not is_us_equity_trading_day(today):
        raise RuntimeError(
            f"Today ({today}) is not a tradable US equity session. Exiting."
        )

    log_path = os.path.join(LOG_DIR, f"s2_paper_{today:%Y%m%d}.txt")
    run_id = str(uuid.uuid4())
    log_line(log_path, f"S2 PAPER RUNNER  fill_date={today}  dry_run={args.dry_run}")
    log_line(log_path, f"  run_id={run_id}")

    strategy = S2Strategy(start_date=str(today))
    panel = strategy.generate_features()
    weights = strategy.get_weights(panel=panel)
    log_line(log_path, "SIGNALS AND WEIGHTS")
    for _, row in weights.iterrows():
        log_line(
            log_path,
            f"  {row['ticker']}  z={row.get('score', float('nan'))}  "
            f"weight={row['weight']}  pair={row.get('pair_id', '')}",
        )
    wsum = float(pd.to_numeric(weights["weight"], errors="coerce").fillna(0.0).sum())
    log_line(log_path, f"  weight_sum={wsum:.6f}  leverage={weights['leverage'].iloc[0] if len(weights) else 1.0}")

    if not args.dry_run:
        log_signals(
            STRATEGY_ID,
            weights,
            broker_id=BROKER_ID,
            run_id=run_id,
            cache_dir=LIVE_LOG_DIR,
        )

    if args.dry_run:
        equity = float(os.environ.get("S2_DRY_RUN_EQUITY", DRY_RUN_EQUITY_DEFAULT))
        current: dict[str, float] = {}
        broker = None
    else:
        if not args.skip_wait:
            ensure_nyse_preopen_submit()
            log_line(log_path, "SUBMIT WINDOW (pre-open / RTH)")
        broker = AlpacaBroker(
            paper=PAPER,
            api_key_name=S2_ALPACA_API_KEY,
            secret_key_name=S2_ALPACA_SECRET_KEY,
        )
        equity = float(broker.get_account().equity)
        current = {
            str(p["ticker"]).strip().upper(): float(p["qty"])
            for p in broker.get_positions_normalized()
        }
        extras = sorted(set(current) - set(strategy.tickers))
        if extras:
            log_line(log_path, f"UNEXPECTED POSITIONS (dedicated account) {extras}")
            for ticker in extras:
                try:
                    broker.close_position(ticker)
                    log_line(log_path, f"  closed {ticker}")
                except Exception as exc:
                    log_line(log_path, f"  close {ticker} failed: {exc}")
            current = {
                str(p["ticker"]).strip().upper(): float(p["qty"])
                for p in broker.get_positions_normalized()
            }

    plans = planned_orders(
        weights,
        equity=equity,
        current=current,
        universe=strategy.tickers,
    )
    log_line(log_path, f"ACCOUNT EQUITY={equity}")
    place_orders(
        broker,
        plans,
        log_path,
        run_id=run_id,
        dry_run=args.dry_run,
        live_log_dir=LIVE_LOG_DIR,
    )
    if broker is not None:
        _log_open_book_snapshot(broker, run_id=run_id, live_log_dir=LIVE_LOG_DIR)


if __name__ == "__main__":
    main()
