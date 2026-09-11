# Universe discovery method

Staged screen implemented in `screen.py` (pre-registered constants — not searched).

1. Group by sector (within-leaf pairs only; nested pools supported)
2. Filter shortability (both legs; Alpaca `shortable`)
3. Filter liquidity (median ADV USD over `ADV_WINDOW`)
4. Reduce load: trailing return corr > `CORR_MIN` (0.90)
5. Engle–Granger + Benjamini–Hochberg FDR at `FDR_Q` (0.10)
6. Round-trip cost ≪ typical `|z|=2` move (`COST_TO_MOVE_MAX` = 0.25)
7. Persistence: trailing `%` ADF p < 0.05 ≥ `PERSISTENCE_FLOOR` (0.40); rolling ADF window `ADF_WINDOW` = 126 (shorter than L so the percentage is defined on a 252-bar residual path)
8. Discovery half-life < `HL_MAX` (40) — HL estimated on the trailing residual path (not the EG scalar when L == OLS window)
9. `select_book` under global / per-pool caps

Trading contract for discovery notebooks (`01_airline_monthly_backtest.ipynb`, `02_tech_monthly_backtest.ipynb`):

- OLS hedge window **252**
- Discovery lookback **252**
- Adaptive z: `clip(2 × HL_{t-1}, 20, 120)`
- Entry `|z| > 2`, exit at mean
- **Monthly** retest at each month-end → effective next session
- Months with **zero** gate survivors are recorded as empty-book markers: no new entries (flat/cash); open trades still run to normal z-exit. The prior month’s pairs are **not** kept silently.
- Airline demo: 50 US-listed airline/carrier candidates (`airline_pools.py`)
- Tech demo: up to **1000** US-listed Technology + Telecommunications names (`tech_pools.load_us_tech_pools`), industry-sharded leaves (max 25); screener CSV cached under `universe_discovery/cache/`
- Full S2 engine: signal close `t` → fill open `t+1`
