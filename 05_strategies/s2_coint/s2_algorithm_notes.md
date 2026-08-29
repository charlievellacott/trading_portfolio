# S2 cointegration — algorithm notes

## Calendar-dense book returns (Sharpe integrity)

When `overlap_mode = never_allow` or a corr gate is active, the joint book simulator (`engine._simulate_book_joint`) used to emit return rows only on days with activity. That dropped flat (cash) days and **inflated Sharpe** vs the `allow` path. Book returns are now reindexed to the panel session calendar via `metrics.book_returns_to_calendar` / `metrics.panel_session_dates` inside `simulate_book`. All fold-val, full-IS, and OOS metrics use the same `n_days`.

## Option 4 research workflow

See `02_research/s2_coint/s2_hypothesis_log.md`. Notebook sequence: `register_hypothesis_arms` → `fold_val_metrics(..., hyp_id=...)` → display `fold_df` + boxplots → `full_is_metrics` → `arm_selection_table(fold_df, full_is_df)` → **type STAR manually** → sealed OOS once. Helpers live in `04_backtest/s2_coint/report.py`; tier map and variant ledger in `04_backtest/s2_coint/research.py`.

## Research inference metrics (PSR / DSR)

**PSR** — `psr = P(true_SR > 1.0 | T, skew, kurt)` on net returns. Probability true Sharpe exceeds 1.0 given sample length T and return shape. Near 1: credible edge at that hurdle; near 0: plausibly luck. Not used for STAR ranking.

**DSR (local)** — `dsr_local` = deflated Sharpe after `N_local` arms in **this** screen. If `dsr_local << ann_sharpe`, the local bake-off may inflate headline Sharpe.

**DSR (stack)** — Same as local but `N_stack` = cumulative arms in `04_backtest/s2_coint/artifacts/s2_variant_ledger.json`. Penalizes sequential H-001 through current hyp search.

**Variant ledger** — Update `s2_variant_ledger.json` via `register_hypothesis_arms(hyp_id, arms, overwrite=True)` when pre-registered **arms add/remove**; not when STAR choice changes. Re-run screens after ledger edits so `dsr_stack` matches. Math: `09_performance/sharpe_inference.py`.

**Not reported:** PBO; `P(true_SR >= reported_SR)` (~0.5 if benchmark equals the point estimate).


## Raw vs adjusted prices (Universe D)

S2 panels fetch **unadjusted** OHLCV (`fetch_ohlcv(..., auto_adjust=False)` in `s2_pair_panel.ipynb`). Adjusted series can phantom-split share-class spreads when one line has dividends and the other does not. β / z / ADF / half-life remain on **close only**; execution still uses opens.

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
- **Pool candidacy:** candidate pairs come from nested pools in `data.processing.s2_universe_pools` (`S2_POOLS`) via `iter_pool_pairs`, which walks **any** nesting depth (universe → exchange → sector → tickers) and forms pairs **only within a leaf pool**. Same-venue is asserted per leaf via `ticker_venue_key`, so a pool that accidentally mixes exchanges raises rather than yielding an unalignable pair. `ticker_venue_key` maps plain US symbols and US share-class lines (`BF.B`, `HEI.A`) to one `US` key so twins are pairable; European suffixes (`.MC`, `.MI`, `.AS`, `.DE`, `.PA`) stay distinct so calendars align. `iter_same_venue_pairs` remains for flat lists.
- **`KEEP_PAIR_IDS` is removed.** The manual keep-list was a discretionary selection channel. The book is now deterministic: Engle-Granger passers ranked by p-value under a **per-pool cap (2)** then a **global cap (6)**, in `strategies.s2_coint.book.select_book`. An unordered pair occupies one slot regardless of orientation.
- Research IS is a **fixed calendar end** per universe, registered in `data.processing.s2_universe_pools.RESEARCH_IS_END_BY_UNIVERSE`: A `2020-12-31`, B `2022-12-31`, C `2021-12-31`, D/E/F `2021-12-31`. Screen and `*_train.parquet` both use `date <= T`; sealed OOS is `date > T`.
- Pair **ineligible** if mutual bars with `date <= T` < `min(ols_window, 252)` — diagnostic NaNs only; not an EG fail; not selected; not re-screened on OOS.
- **Two-pass panel build** (`s2_pair_panel.ipynb`): pass 1 screens on **closes only** (no panel); pass 2 builds panels with per-bar `adf_pvalue` for the **union of pairs ever selected**. Rolling ADF exists solely for the `BREAK_STAR` health gate on an actively traded pair, so materialising panels for candidates that no point-in-time rebalance ever selects would change no signal. `BOOK_SCOPE` controls the union: `freeze_only` for H-001, `freeze_plus_rotate` for H-004.
- Panel traditional z uses fixed `Z_WINDOW=60`. Rolling `half_life` (`HL_WINDOW=252`) is a **diagnostic only** until H-008 — no half-life filter exists anywhere before it. Adaptive z-window is **H-014** and is not used in panel v1.
- Discovery EG p-value and discovery half-life live on `s2_pairs_*_1d.csv` (IS-only). They do not set panel z.
- Fetch OHLCV in the notebook (or live runner); `s2_coint_store.build_pair_panel` takes per-ticker OHLC frames and is timeframe-agnostic.
- Panel columns: `open_y/high_y/low_y/close_y` and `open_x/high_x/low_x/close_x` (no `price_y` / `price_x`). Hedge / z / ADF / half-life use **closes only**; open/high/low are for fill/stop/PnL.
- Outputs under `01_data/data_files/s2_coint/`: `s2_panel_{letter}_1d_{train,full}.parquet` (union pairs), `s2_pairs_{letter}_1d.csv` (ranked frozen book), `s2_candidates_{letter}_1d.csv` (every candidate at IS end, so rejects stay auditable), and `s2_rebalance_{letter}_1d.csv` (quarterly selections, rotate scope only).

## Book construction (H-004)

- Bake-off is `freeze` (one screen on the **full IS** at `RESEARCH_IS_END`) vs `rotate` (quarterly re-screen on the trailing `L = 252` bars, effective at the **next** session's open). Both arms read one identical union panel and share the health gate, soft defaults, caps, beta hedging and slot weighting, so the only difference is book construction. Rules live in `strategies.s2_coint.book`; the research driver is `backtest.s2_coint.rotation`.
- **Pre-registered constants, not searched and not STARs:** quarterly cadence, `L = 252`, `alpha = 0.05`, global cap 6, per-pool cap 2, fixed `1/6` slot weighting, no minimum tenure, no half-life filter before H-008.
- **H-004 only, and continues if `BOOK_STAR = rotate`:** a demoted pair is blocked from **new** entries but runs any open position to its normal z-exit. Nothing is force-closed, so rotation adds no trading cost of its own. Orientation flips are handled the same way — the old orientation is blocked and runs to z-exit, and the new one becomes tradable only once the old is flat.
- **Sizing:** beta hedging is unchanged (`+y, -beta*x`, gross-normalised). The `1/6` slot weight multiplies that already-normalised position, so a one-pair quarter cannot run six times the per-pair risk of a six-pair quarter and confound the comparison.
- **Orientation:** Engle-Granger tests both `y~x` and `x~y` and keeps the lower p-value; the winning direction sets `pair_id`. Orientation is **frozen while a pair is active** and only re-evaluated if the pair is demoted and later re-promoted.
- **Metrics:** report `ann_sharpe_gross` and `ann_sharpe_net` (gross via `gross_returns_from_net`, not a second cost-free sim), max DD, `corr_to_s1`, and book composition by pool. Rank on net Sharpe; **a negative correlation to S1 beats a low positive one**.

## Short-selling bans

- Regulator ban windows live in `strategies.s2_coint.short_bans` (`SHORT_BANS`). They block **new entries only** in the spread direction that needs the banned leg: long spread shorts `x`, short spread shorts `y`. Open positions still exit on z, mirroring real bans, which restricted opening or increasing net short positions rather than forcing liquidation.
- Mechanically this is the same mask as demotion, composed by logical AND, via the optional `long_entry_allowed` / `short_entry_allowed` arguments on `engine.simulate_pair`. With both masks absent, behaviour is unchanged.
- Applies to **universe F only** (Madrid / Milan / Paris in 2011–12 and 2020). A–E have no records, so their masks are all-True and their results are byte-identical with or without them. Amsterdam and Xetra were never banned (AFM and BaFin declined in 2020).
- Not modelled: market-maker exemptions, the 2020 net-short thresholds, and the SEC's 2008 US ban (list membership uncertain for REITs and share classes). **Short borrow is still not modelled** in any cost profile.

## H-001 baseline costs / IS diagnostics

- Source of truth: `strategies.s2_coint.costs` (`COSTS`, routing, `leg_cost_bps`). Simulator: `strategies.s2_coint.baseline`. Per-pair IS tables: `strategies.s2_coint.metrics`.
- H-001 notebook (`02_research/s2_coint/notebooks/hypothesis_tests/H-001_universes.ipynb`) calls those modules. Evaluation is **research IS only** (OHLC clipped to `RESEARCH_IS_END` before the pair panel is built).
- Costs are modeled per leg at entry and exit via `COSTS`:
  - `A_FX_OANDA`: spread+slippage model with pair-level spread pips.
  - `B_CRYPTO_KRAKEN`: maker/taker + slippage model (baseline assumes taker).
  - `C_HK_IBKR` / `C_JP_IBKR`: percent commission + minimum + third-party fee + slippage.
  - `US_ALPACA` (universes D / E): **commission-free**, so cost is regulatory fees (SEC Section 31 + FINRA TAF, 0.1 bps) plus 3.2 bps execution → **3.3 bps/leg, ~13 bps per pair round trip**. Execution cost is broker-agnostic slippage against a VWAP benchmark, and the upper end of the published band is used deliberately because a pair fills two legs at once and one is a short. Roughly **9x less friction than C's HK 116 bps/RT**, which is the quantitative case for the universe pivot.
  - `US_ALPACA_D_REALISTIC` (Universe D default via `config_from_stack`): common leg **3.2 bps** slippage; alt share-class lines (`.A`, `.B`, `NWSA`) **8.0 bps**; **100 bps/year** borrow pro-rated daily on net short leg weight. **Not modelled:** vol-scaled slippage, locate fees, HTB spikes, pair-specific spread — document as stress-only fairness bounds in research notes.
  - `F_EUR_IBKR` (universe F): percent commission + EUR minimum + third-party + slippage → ~10.5 bps/leg. **This is an assumption**, not a confirmed schedule: IBKR's published Reg-NMS metrics describe US stocks and the 0.1% / EUR 4 / EUR 29 table is the mutual-fund schedule, so neither applies to EUR cash equities. Confirm Fixed vs Tiered and per-exchange minimums for Madrid / Milan / Amsterdam / Xetra / Paris before treating F's net Sharpe as decision-grade; F is the most cost-fragile universe.
- Market routing is deterministic:
  - `=X` -> `A_FX_OANDA`
  - `-USD` -> `B_CRYPTO_KRAKEN`
  - `.HK` -> `C_HK_IBKR`
  - `.T` -> `C_JP_IBKR`
  - `.MC` / `.MI` / `.AS` / `.DE` / `.PA` -> `F_EUR_IBKR`
  - plain US symbols and US share-class lines (`GOOGL`, `AMT`, `BF.B`) -> `US_ALPACA`
- Baseline sizing uses hedge ratio (`beta`) per bar: long spread `+y, -beta*x`; short spread `-y, +beta*x`. Trad-z exit is a signed recross of `EXIT_Z` (default 0): flatten a long spread when `z >= 0`, a short when `z <= 0`.
- Per-pair IS stats: trade count, median hold (completed round-trips), cost bps/year, Sharpe, max DD, rolling ADF (`adf_pvalue` from `compute_coint_metrics`).
- Asia C IS postmortem → `02_research/s2_coint/notebooks/other_tests/01_asia_c_failure_diagnosis.ipynb` (helpers in `04_backtest/s2_coint/diagnosis.py`).
- **Universe C shelved.** Gross Sharpe ≈ 0 (+0.02 to +0.24) before costs, net −0.08 to −0.48 after. Cost drag 271–439 bps/yr (HK 116 bps/RT, JP 64 bps/RT); median rolling ADF p 0.19–0.30, significant only 11–29% of days. Not a timing bug and not a trade-frequency problem (~3.6–4.2 round-trips/yr, `|z|>2` on ~12% of days). Per-pair table in `02_research/s2_coint/universe.md`; archived artifacts in `04_backtest/s2_coint/artifacts/asia_c/`.
- Overlay check: compound IS daily book returns to S1 Monday–Monday weeks and correlate vs `01_data/data_files/s1_equities/s1_period_returns.parquet` (exported from `08_oos_tearsheet.ipynb`). Missing file → `corr_to_s1` is NaN.

## Live paper (hardcoded STAR)

Frozen recipe is `04_backtest/s2_coint/artifacts/s2_star_stack.json` (see `01_star_tearsheet.ipynb`). Paper runner: `07_execution/s2_coint/s2_paper_runner.py`. Dedicated Alpaca paper account (100% of that account equity). Credentials: `config/credentials_s2.env` (`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`), or `ALPACA_S2_API_KEY` / `ALPACA_S2_SECRET_KEY` in `config/credentials.env`, or `S2_ALPACA_CREDENTIALS`. **Never falls back to S1 keys.** Logs: `07_execution/s2_coint/logs/s2_paper_YYYYMMDD.txt` (not the S1 log dir). Live ledger: `09_performance/cache/live_s2/`. Cache: `05_strategies/s2_coint/cache/` (`S2_CACHE_DIR` override).

Clock: fill morning of `t+1`. Features from last completed close `t` (drop any `date >= fill_date`). Wait until 09:28 ET, then DAY market deltas (no resting stops; STAR `EXIT_STAR=mean_only`). `--dry-run` prints orders and does not submit.

**Score sizing denominator (live, not in the STAR tearsheet):** `mean_abs_score` is the per-pair rolling mean of `|z|` with `window = Z_WINDOW_STAR` (frozen **90**, not the H-001 panel default 60) and `min_periods = window`, through close `t`. The sealed backtest (`01_star_tearsheet.ipynb`) instead freezes the **research-IS mean of `|z|`** via `fit_mean_abs_score` and does not refit on OOS. Rolling live scale was **not** walk-forwarded — backtest it before treating live sizing as research-parity.

## Notes
- S1 returns are computed weekly and so the correlation was calculated weekly 