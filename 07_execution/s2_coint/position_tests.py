from execution.brokers.alpaca_broker import AlpacaBroker, S2_ALPACA_API_KEY, S2_ALPACA_SECRET_KEY

b = AlpacaBroker(paper=True, api_key_name=S2_ALPACA_API_KEY, secret_key_name=S2_ALPACA_SECRET_KEY)

for ticker in ["NWSA", "WSO.B", "NWS", "WSO"]:
    a = b.client.get_asset(ticker)
    print(ticker, "tradable=", a.tradable, "shortable=", getattr(a, "shortable", None),
          "status=", a.status, "class=", getattr(a, "asset_class", None))

# Tiny probes (flatten after). Same order as the runner: sells then buys.
for ticker, side, qty in [("NWSA", "sell", 1), ("WSO.B", "sell", 1), ("NWS", "buy", 1), ("WSO", "buy", 1)]:
    print("SUBMIT", side, qty, ticker, flush=True)
    try:
        o = b.submit_order(ticker=ticker, quantity=qty, size=None, direction=side)
        print("  OK", o.id, o.status, flush=True)
    except Exception as e:
        print("  FAIL", type(e).__name__, e, flush=True)
        break