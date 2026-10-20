# S2 v2 universe

Point-in-time S&P 500 names from `data.ingestion.sp500_universe.constituents_on_or_before`.
Pairs form only inside the same industry. Prices from `data.ingestion.equity_fetcher.fetch_ohlcv`.

Research IS, sealed holdout, and the forward paper window are frozen in H-001 once the panel span is known. Do not use v1's letter-universe end date as this generation's cut.

Survivorship: names without yfinance history are dropped and the drop rate is reported. They are not filled.
