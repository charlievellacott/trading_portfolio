# S2 v2 universe

Point-in-time S&P 500 names from `data.ingestion.sp500_universe.constituents_on_or_before`.
Prices from `data.ingestion.equity_fetcher.fetch_ohlcv` (`auto_adjust=False`).
GICS sector / industry from `data.ingestion.equity_gics.fetch_yahoo_gics` (**current** Yahoo labels, not historical). Document that classification look-ahead.

Pairs form only inside the same leaf after H-001 freezes `POOL_STAR` (`sector` or `industry`).
No unrestricted cross-sector arm.

Research IS, sealed holdout, and `PAPER_START` are frozen in
`01_data/data_files/s2_coint/v2/s2_pair_panel.ipynb` Pass A (`s2_cuts.json`).
Do not use v1's letter-universe end date as this generation's cut.

Survivorship: names without yfinance history are dropped and the drop rate is reported. They are not filled.
Liquidity: ADV floor at the research cut; report the share dropped.

Panel artifacts (no `pit`, no universe letter) live under `01_data/data_files/s2_coint/v2/`:
`s2_candidates_1d.csv`, `s2_panel_1d_{train,full}.parquet`, `s2_rebalance_1d.csv`.
