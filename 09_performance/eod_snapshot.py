"""Daily EOD live performance snapshot (stop-fill sync + mark-to-market)."""

# Imports
import argparse

from execution.brokers.alpaca_broker import AlpacaBroker
from performance.live_log import run_eod_snapshot

# Constants
DEFAULT_PAPER = True


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync stop fills and write live_log EOD equity/positions."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live Alpaca account instead of paper (default: paper).",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Upsert daily equity from Alpaca portfolio history before today's mark.",
    )
    parser.add_argument(
        "--backfill-period",
        default="1M",
        help="Alpaca history period for --backfill (default: 1M).",
    )
    args = parser.parse_args()
    paper = not args.live
    broker = AlpacaBroker(paper=paper)
    result = run_eod_snapshot(
        broker,
        paper=paper,
        backfill=bool(args.backfill),
        backfill_period=str(args.backfill_period),
    )
    print(
        f"EOD snapshot date={result['date'].date()} "
        f"portfolio_equity={result['portfolio_equity']:.2f} "
        f"stop_fills_added={result['stop_fills_added']} "
        f"backfill_rows_delta={result.get('backfill_rows_delta', 0)}"
    )


if __name__ == "__main__":
    main()
