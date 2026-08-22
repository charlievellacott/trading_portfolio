# S2 cointegration — algorithm notes

## Timing contract

- Features / signal / decision at **close of bar `t`** (`close_y` / `close_x`).
- Optional alt data after close `t` and before open `t+1` if knowable then; orders queued just before open `t+1`.
- Fill at **open of `t+1`** (both legs; no-auction venues: next-bar OHLCV open). First PnL from open `t+1` onward.
- Same causal lag at every bar size (1D / 4H / 1H — H-002).
- Not Default same-bar close-fill; not S1 trade-date / `feature_date` indexing.
- Do not add an extra `.shift(1)` on close-`t` features; do not store next-bar open on the signal row.
- Backtest stops: path-check high/low after entry. Live/paper: resting stop/limit after fill.

## Panel / universe

- No global `START_DATE` for the pair panel: each pair’s history starts on the first date both legs have a valid close. Early calendar dates can omit pairs whose younger leg is missing; that is intentional (unbalanced panel).
- Candidate pairs come from `data.processing.s2_universe` (`ticker_venue_key` / `iter_same_venue_pairs`): same venue/suffix only (e.g. `.HK` with `.HK`, `.T` with `.T`, FX `=X`, crypto `-USD`). Within-venue theme filters (e.g. HK banks vs HK oil SOEs) are applied via `SECTOR_PAIR_MAP` / `KEEP_PAIR_IDS` in `H-001_universes.ipynb`. C refined: China banks, JP megabanks, China oil SOEs (no tech/semis; no bank↔oil).
- Research IS is a **fixed calendar end** per universe in `s2_pair_panel.ipynb`: A `2020-12-31`, B `2022-12-31`, C `2021-12-31`. Screen and `*_train.parquet` both use `date <= T`; sealed OOS is `date > T`.
- Pair **ineligible** if mutual bars with `date <= T` < `min(ols_window, 252)` — diagnostic NaNs only; not an EG fail; not locked; not re-screened on OOS.
- Final locked book = eligible research-IS Engle–Granger passers (`pvalue < threshold`) + `KEEP_PAIR_IDS`. Do not re-pick pairs after seeing sealed OOS.
- Panel traditional z uses fixed `Z_WINDOW=60`. Rolling `half_life` (`HL_WINDOW=252`) is for gates / time-stops (H-006 / H-008). Adaptive z-window is **H-012** and is not used in panel v1.
- Discovery EG p-value and discovery half-life live on `s2_pairs_*_1d.csv` (IS-only). They do not set panel z.
- Fetch OHLCV in the notebook (or live runner); `s2_coint_store.build_pair_panel` takes per-ticker OHLC frames and is timeframe-agnostic.
- Panel columns: `open_y/high_y/low_y/close_y` and `open_x/high_x/low_x/close_x` (no `price_y` / `price_x`). Hedge / z / ADF / half-life use **closes only**; open/high/low are for fill/stop/PnL.
- Outputs: `s2_panel_{A,B,C}_1d_{train,full}.parquet` and `s2_pairs_{A,B,C}_1d.csv` under `01_data/data_files/s2_coint/`.

## H-001 baseline costs / IS diagnostics

- Source of truth: `strategies.s2_coint.costs` (`COSTS`, routing, `leg_cost_bps`). Simulator: `strategies.s2_coint.baseline`. Per-pair IS tables: `strategies.s2_coint.metrics`.
- H-001 notebook (`02_research/s2_coint/notebooks/hypothesis_tests/H-001_universes.ipynb`) calls those modules. Evaluation is **research IS only** (OHLC clipped to `RESEARCH_IS_END` before the pair panel is built).
- Costs are modeled per leg at entry and exit via `COSTS`:
  - `A_FX_OANDA`: spread+slippage model with pair-level spread pips.
  - `B_CRYPTO_KRAKEN`: maker/taker + slippage model (baseline assumes taker).
  - `C_HK_IBKR` / `C_JP_IBKR`: percent commission + minimum + third-party fee + slippage.
- Market routing is deterministic:
  - `=X` -> `A_FX_OANDA`
  - `-USD` -> `B_CRYPTO_KRAKEN`
  - `.HK` -> `C_HK_IBKR`
  - `.T` -> `C_JP_IBKR`
- Baseline sizing uses hedge ratio (`beta`) per bar: long spread `+y, -beta*x`; short spread `-y, +beta*x`. Trad-z exit is a signed recross of `EXIT_Z` (default 0): flatten a long spread when `z >= 0`, a short when `z <= 0`.
- Per-pair IS stats: trade count, median hold (completed round-trips), cost bps/year, Sharpe, max DD, rolling ADF (`adf_pvalue` from `compute_coint_metrics`).
- Asia C IS postmortem → `02_research/s2_coint/notebooks/other_tests/01_asia_c_failure_diagnosis.ipynb` (helpers in `04_backtest/s2_coint/diagnosis.py`).
- Overlay check: compound IS daily book returns to S1 Monday–Monday weeks and correlate vs `01_data/data_files/s1_equities/s1_period_returns.parquet` (exported from `08_oos_tearsheet.ipynb`). Missing file → `corr_to_s1` is NaN.

## Notes
- S1 returns are computed weekly and so the correlation was calculated weekly 