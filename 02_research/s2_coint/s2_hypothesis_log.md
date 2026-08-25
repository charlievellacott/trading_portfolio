# Hypothesis Log

## Validation protocol

Mirror `03_models/s1_equities/model_tests/06_training_gbm_lambdarank_wf.ipynb`: purged expanding walk-forward on **research IS** → freeze by selection score + **boxplot review across folds** → **one** sealed OOS look for tearsheet / final backtest. Embargo between fold-train and fold-val (pairs analogue of the GBM 1-week embargo).

- **Fold-train (pairs):** estimate / warm anything that must be fit from past data inside the strategy (pair formation, PIT β / Kalman state, half-life estimate, OU/ADF inputs, corr KF state). Do **not** pick discrete design knobs by maximising train Sharpe.
- **Fold-val:** score each **pre-registered** candidate config (including knobs like ATR multiple) on that fold’s validation segment. Aggregate across folds (boxplots of Sharpe, max DD, corr to S1). Freeze the winner on research IS only; never tune on sealed OOS.
- **Research IS / sealed OOS:** research IS is the sleeve train panel through that universe’s fixed `RESEARCH_IS_END` (A `2020-12-31`, B `2022-12-31`, C `2021-12-31`, **D / E / F** `2021-12-31`); sealed OOS is after it. Registered in `data.processing.s2_universe_pools.RESEARCH_IS_END_BY_UNIVERSE`. Under `BOOK_STAR = freeze` the pair list is frozen at discovery on `date <= T`; under `rotate` it is re-selected each quarter on data `<= T_rebalance` only (H-004).
- **Primary metrics (every hyp):** net **Sharpe**, **max drawdown**, and **correlation to S1**. Rank on net Sharpe and report correlation beside it — **no combined score**. Read correlation directionally: **a negative correlation to S1 is strictly better than a low positive one** (−0.2 beats 0.0), because negative correlation adds diversification rather than merely avoiding overlap.



### Timing contract (all S2 hyps / all bar sizes)

Not Default same-bar close-fill; not S1 trade-date / `feature_date` indexing.


| Role                         | Timestamp                                                         |
| ---------------------------- | ----------------------------------------------------------------- |
| Features / signal / decision | Close of bar `t`                                                  |
| Optional alt data            | After close `t`, before open `t+1` (if knowable then)             |
| Orders                       | Queued just before open `t+1`                                     |
| Fill                         | Open of `t+1` (both legs; no-auction venues: next-bar OHLCV open) |
| First PnL / HL stops         | From open `t+1` onward                                            |


Do **not** add an extra `.shift(1)` on close-`t` features. Do **not** materialize next-bar open on the signal row as a feature. Same lag for 1D / 4H / 1H (H-002).

---



## Notes

> **Spread history vs β (must confirm):** When building a rolling spread series for z-scores, half-life, ADF, etc., does s_{t-k} use **β at that past time** \beta_{t-k} (point-in-time: s_{t-k} = y_{t-k} - \beta_{t-k} x_{t-k} - \alpha_{t-k}), or does it reuse the **current** β \beta_t on past prices (look-ahead / revision of history)?
>
> **Default for this sleeve:** use **β (and α) as of each timestamp** — never rewrite past spreads with today’s β. Kalman paths emit \beta_t each bar; static OLS must use only information available at that bar (rolling/expanding window ending at t-k), not a full-sample β.



### Implementation conventions (math / store)

- **Public API:** `01_data/processing/s2_coint_store.py` (`compute_`* / `run_cointegration_test` / `build_pair_panel`). Math lives in `feature_implementation/cointegration.py`; shared KF core in `feature_implementation/kalman.py`. Universe candidacy: `01_data/processing/s2_universe.py`.
- **Panel schema:** row `date = t` is the signal bar. Columns `open_y/high_y/low_y/close_y` and `open_x/high_x/low_x/close_x` (no `price_y` / `price_x`). Hedge, z, ADF, half-life use `close_y` **/** `close_x` **only**. Open/high/low are for fill/stop/PnL path — never inputs to β. Fill for a signal on `t` = next pair-bar `open_`* (lookup at backtest time; do not store `shift(-1)` open on the signal row).
- **Log prices:** store converts raw closes via `to_log_price`; math expects logs. Spread: s_t = y_t - \alpha_t - \beta_t x_t on closes.
- **Engle-Granger discovery:** test both `y~x` and `x~y`, keep lower p-value; separate OLS for α/β. `COINT_PVALUE = 0.05` flat — EG has low finite-sample power on small pre-specified lists; holdout confirmation is the real gate (no Bonferroni on a handful of economic pairs).
- **Bayesian / MCMC β:** deferred. Full-sample MCMC / smoother paths rewrite history with future info and violate PIT; if revisited, calibrate noise on fold-train only, then run the recursive filter live.

**Traditional z baseline:** H-006 and H-008 use fixed-k rolling z-score entry/exit (default `Z_WINDOW=60` unless `Z_WINDOW_STAR` frozen in H-005). Adaptive z-window is H-014. Extremity-score alternatives are H-015.

**Pool candidacy:** candidates come from nested pools in `01_data/processing/s2_universe_pools.py` (`S2_POOLS`) via `iter_pool_pairs`, which walks **any** nesting depth and forms pairs **only within a leaf pool**. Same-venue is asserted per leaf, so a pool mixing exchanges raises instead of producing an unalignable pair. Live / rotate book construction (H-004) is deterministic p-value ranking under caps via `select_book`. H-001's freeze book is a **documented** `MANUAL_BOOK` under the same caps (`validate_manual_book`) — not a silent keep-list.

**Pre-registered book constants (not searched, not STARs):** quarterly rebalance, `L = 252` trailing bars for the rotating screen, `alpha = 0.05`, global cap **6**, per-pool cap **2**, fixed `1/6` slot weighting. No half-life filter anywhere before H-008. No minimum tenure. Beta-hedged legs throughout.

**Short-selling bans:** regulator windows in `05_strategies/s2_coint/short_bans.py` block **new entries only** in the spread direction that needs the banned leg (long spread shorts `x`, short spread shorts `y`); open positions still exit on z. Universe F only — A–E have no records, so their masks are all-True. Short borrow remains unmodelled.

**Shared conflict-resolution rules** (H-009 never-allow arm; H-011 when |\hat\rho_t| > k):

1. If a position is **already open** and a new candidate **conflicts** with it → **do not open** the new position.
2. If **both** candidates would open on the **same bar** → open only the one with the best **Score × confidence** (|\text{score}| \times (1 - p_{\text{ADF}}); score from the active entry rule / chosen H-015 variant when in use).

---


| ID    | Date       | Asset | Factor                                          | Data required | Status                                      |
| ----- | ---------- | ----- | ----------------------------------------------- | ------------- | ------------------------------------------- |
| H-001 | 2026-08-10 | Coint | Universes A–F                                   | —             | DECIDED: C = Shelved D = Accepted (for now) |
| H-002 | 2026-08-10 | Coint | 1D vs 4H / 1H after costs                       | —             | DECIDED: 1D                                 |
| H-003 | 2026-08-10 | Coint | Cointegration-break flat rule (Part A)          | —             | DECIDED: block_05_flat_10                   |
| H-004 | 2026-08-23 | Coint | Book construction: frozen vs quarterly rotating | —             | NOT IMPLEMENTED                             |
| H-005 | 2026-08-23 | Coint | Window screens (OLS / ADF / entry z / z)        | —             | NOT IMPLEMENTED                             |
| H-006 | 2026-08-10 | Coint | Kalman β vs static OLS hedge (trad z)           | —             | NOT IMPLEMENTED                             |
| H-007 | 2026-08-10 | Coint | ADX and/or RSI trend filter                     | —             | NOT IMPLEMENTED                             |
| H-008 | 2026-08-10 | Coint | Half-life gate (trad z)                         | —             | NOT IMPLEMENTED                             |
| H-009 | 2026-08-10 | Coint | Overlapping legs: allow vs never-allow          | —             | NOT IMPLEMENTED                             |
| H-010 | 2026-08-10 | Coint | Exit: n half-lives + ATR SL + max-loss breaker  | —             | NOT IMPLEMENTED                             |
| H-011 | 2026-08-10 | Coint | Kalman spread-correlation gate                  | —             | NOT IMPLEMENTED                             |
| H-012 | 2026-08-10 | Coint | Score × confidence sizing                       | —             | NOT IMPLEMENTED                             |
| H-013 | 2026-08-10 | Coint | Vol-aware entry k_t vs S1 vol targeting         | —             | NOT IMPLEMENTED                             |
| H-014 | 2026-08-12 | Coint | Adaptive z-window (trad z)                      | —             | NOT IMPLEMENTED                             |
| H-015 | 2026-08-10 | Coint | Entry extremity scores                          | —             | NOT IMPLEMENTED                             |


**Renumbering (2026-08-23).** Old H-004…H-013 shifted **+2** to H-006…H-015. Old H-003 split: Part A stayed at H-003 (break rule), Part B became **H-005** (window screens), and the new **H-004** (book construction) sits between them. The +2 shift is deliberate: fitting the window screens *before* the book bake-off would tune them on the frozen architecture and bias H-004 toward `freeze`.

## H-001 · Coint · Universes A / B / C · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **What it is**         | Compare candidate pair universes **A–F** under the same trading rules. A (FX), B (crypto), C (Asia EM) are re-run for the record only; **D** (US share-class twins), **E** (US REIT sub-sectors), **F** (EUR large caps) compete for `UNIVERSE_STAR`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Hypothesis**         | At least one of D / E / F delivers positive net Sharpe after costs with low or negative correlation to S1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Economic rationale** | D pairs claims on the **same** cash flows (share classes); E pairs same-subsector REITs sharing a cap-rate / rates factor; F pairs same-exchange, same-sector EUR large caps. All three sit on venues with far lower friction than C (US ~3.3 bps/leg vs HK 29 bps/leg), which is the binding constraint that killed C.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Data required**      | Daily OHLCV per pool ticker (Yahoo via `fetch_ohlcv`; `isAsian=True` for exchange-suffixed names).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Test to complete**   | Validation protocol: WF fold-val boxplots of Sharpe, max DD, corr to S1; freeze universe on research IS; one sealed OOS tearsheet.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Notes**              | Lock chosen universe before later hyps; do not re-pick after downstream bake-offs. Notebook `H-001_universes.ipynb` runs (i) pool-scoped EG screening via `iter_pool_pairs`, (ii) inspect **all** EG passers with `annotate_screen` / trad-z / `pct_adf_lt_threshold` / ADF overlap, (iii) documented `MANUAL_BOOK` freeze under caps 6 / 2 (`validate_manual_book` — not a silent keep-list; auto `select_book` is a suggestion only), (iv) book Sharpe on the locked books. Scores each universe under the `freeze` convention with soft defaults, since H-003 / H-004 have not run yet (no break gate). Baseline constants (`ENTRY_Z=2`, `EXIT_Z=0`, beta-sized legs) and market costs (OANDA FX, Kraken spot, IBKR HK/JP, **Alpaca US**, **IBKR EUR**) live in `strategies.s2_coint`. Reports net Sharpe and `corr_to_s1` side by side — negative correlation beats low positive. Writes `s2_pairs_{letter}_1d.csv`. **Research IS only**; do not use sealed OOS to pick the universe. Asia C IS postmortem → `02_research/s2_coint/notebooks/other_tests/01_asia_c_failure_diagnosis.ipynb`. **F cost caveat:** `F_EUR_IBKR` is an assumption — confirm IBKR's European cash-equity schedule before scoring F. |




**Universe C shelved (2026-08).** Gross Sharpe ≈ 0 (+0.02 to +0.24) before costs, net −0.08 to −0.48 after. Cost drag 271–439 bps/yr (HK 116 bps/RT, JP 64 bps/RT); median rolling ADF p 0.19–0.30, significant only 11–29% of days. Not a timing bug and not a trade-frequency problem (~3.6–4.2 round-trips/yr, `|z|>2` on ~12% of days). Per-pair table lives in `02_research/s2_coint/universe.md`; archived Asia artifacts in `04_backtest/s2_coint/artifacts/asia_c/`. A / B / C stay documented as learning points and are not re-selected.

---



## H-002 · Coint · 1D vs 4H / 1H after costs · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                          |
| **What it is**         | Same pair logic on daily vs 4H vs 1H bars, with timeframe-appropriate frictions.                                                                                                                                                                                                         |
| **Hypothesis**         | 1D timeframe survives costs better than 4H/1H after frictions.                                                                                                                                                                                                                           |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                        |
| **Data required**      | —                                                                                                                                                                                                                                                                                        |
| **Test to complete**   | Compare on Sharpe, max DD, corr to S1 under the Validation protocol (identical pair set and entry rule where possible).                                                                                                                                                                  |
| **Notes**              | Same Timing contract at **1D / 4H / 1H** (signal close `t` → fill open `t+1`). Compare on Sharpe, max DD, corr to S1 under the Validation protocol. **Bar only** — OLS / ADF / entry / z screens live in H-005 under frozen `BREAK_STAR` (Pipeline A). Notebook: `H-002_bar_size.ipynb`. |


---



## H-003 · Coint · Cointegration-break flat rule · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                |
| **What it is**         | Go flat / block when cointegration or stationarity breaks on the live spread. Window screens moved out to **H-005** so they are fitted after the book decision, not before it.                                                                                                                                                                                                                                                 |
| **Hypothesis**         | Cointegration-break flat rule improves max DD (and Sharpe / corr to S1) more than it hurts return.                                                                                                                                                                                                                                                                                                                             |
| **Economic rationale** | Universe C's spread was non-stationary most of the time (median rolling ADF p 0.19–0.30), so an explicit health gate should stop trading a dead relationship rather than paying costs to fade noise.                                                                                                                                                                                                                           |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Test to complete**   | Validation protocol: break arms on fold-val boxplots → sleeve kill check → freeze `BREAK_STAR`; sealed OOS once.                                                                                                                                                                                                                                                                                                               |
| **Notes**              | **Impl health metrics** via `compute_coint_metrics`: `adf_pvalue` (rolling ADF on PIT spread) and `variance_jump`. Discovery EG stays out of the live loop. Soft defaults: ols/adf 252d, entry_z 2, z 60 — these stay soft here and are only screened in H-005. Runs on the **frozen** book so the gate is isolated from book construction. STAR: `BREAK_STAR`. Short borrow not modeled. Notebook: `H-003_coint_break.ipynb`. |


**Sleeve kill / continue (before H-004).** Metric = median fold-val `ann_sharpe` (net). Do not use sealed OOS.


| Band            | Median fold-val Sharpe | Action                                                                                                                                 |
| --------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Severe          | **≤ −0.50**            | Shelve sleeve — no H-004 / H-005+                                                                                                      |
| Deeply negative | **(−0.50, −0.25]**     | Shelve unless beats `off` by **≥ +0.25** Sharpe **or** median max DD **≥ 5 pp** better **and** `pct_adf_lt_0.05` **≥ +10 pp** vs `off` |
| Soft negative   | **(−0.25, 0]**         | Yellow — continue only if break clearly helped; document why                                                                           |
| Non-negative    | **> 0**                | Continue to H-004                                                                                                                      |


---



## H-004 · Coint · Book construction: frozen vs quarterly rotating · 2026-08-23


| Field                  |                                                                                                                                                                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                             |
| **What it is**         | Bake-off of how the pair book is chosen: `freeze` (one-shot lock at `RESEARCH_IS_END`) vs `rotate` (quarterly re-screen, promote / demote inside industry pools). Everything else is held identical.                                                                                        |
| **Hypothesis**         | A quarterly rotating book beats a frozen book on net Sharpe (max DD / corr to S1 secondary), because relationships decay and rotation drops dead pairs while admitting newly cointegrated ones.                                                                                             |
| **Economic rationale** | Universe C showed EG passing once at discovery while rolling ADF stayed insignificant 71–89% of the time. Freezing keeps trading a dead relationship; rotation is the direct structural fix. The counter-risk is selection noise from repeated screening, which the caps and ranking bound. |
| **Data required**      | Union panel (§ pass 2) plus quarterly EG screens on closes.                                                                                                                                                                                                                                 |
| **Test to complete**   | Validation protocol: both arms on identical folds → fold-val boxplots of Sharpe, max DD, corr to S1 → freeze `BOOK_STAR`; sealed OOS once.                                                                                                                                                  |
| **Notes**              | Impl: `strategies.s2_coint.book` (selection, caps, `BookState`) and `backtest.s2_coint.rotation` (schedule, both simulators). Notebook: `H-004_book_construction.ipynb`.                                                                                                                    |


**Arms.** Both share the union panel, `BREAK_STAR`, soft defaults, caps, beta hedging and slot weighting.


|                    | `freeze`                                    | `rotate`                   |
| ------------------ | ------------------------------------------- | -------------------------- |
| Screen date(s)     | once at `RESEARCH_IS_END`                   | every quarter-end session  |
| Screen window      | **full IS** (`date <= T`)                   | trailing `L = 252` bars    |
| Book               | fixed for the whole evaluation              | re-selected each rebalance |
| α / ranking / caps | 0.05, lowest p-value, 6 global & 2 per pool | identical                  |


The window asymmetry is deliberate: full IS *is* what freezing means here ("cointegrated in general"), while the trailing year is what rotation means ("cointegrated recently"). Forcing both to `L = 252` would handicap `freeze`.

**Rotating rules (pre-registered, not searched).**

- Rebalance on the last session of Mar / Jun / Sep / Dec; the new active set is effective from the **next session's open**.
- While active, the ADF health gate is `BREAK_STAR` from H-003.
- **Demotion:** blocked from **new** entries; any open position runs to its normal z-exit. Nothing is force-closed, so rotation adds no trading cost of its own.
- **Orientation flip:** treated as demote-then-promote. The old orientation is blocked and runs to z-exit; the new one waits until the old is flat. An unordered pair occupies **one** slot either way, so caps never double-count.
- **No min tenure** (redundant once demotion forces no trade) and **no half-life filter** (deferred to H-008).
- Short-ban masks apply in both arms and compose with the demotion mask by logical AND.

**Sizing.** Beta hedging unchanged (`+y, -beta*x`, gross-normalised). On top, a fixed `1/6` **per slot**, cash otherwise, in both arms — so a one-pair quarter cannot run six times the per-pair risk of a six-pair quarter and confound the comparison.

**Reported metrics.** `ann_sharpe_gross`, `ann_sharpe_net`, max DD, `corr_to_s1`, active pairs per quarter, promotions / demotions, and **book composition by pool** (surfaces the single-factor concentration that sank C). `cost_bps_year` is dropped: the gross-vs-net gap says the same thing more directly. Gross reuses `gross_returns_from_net` in `diagnosis.py` rather than a second cost-free simulation.

**Continue / shelve.** Winner = higher median fold-val Sharpe. If the **winning** arm's median fold-val Sharpe ≤ 0, shelve that universe and return to the next-ranked H-001 universe instead of proceeding to H-005.

---



## H-005 · Coint · Window screens (OLS / ADF / entry z / z) · 2026-08-23


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **What it is**         | Sequential screens for OLS lookback, ADF window, entry z and fixed z-window, under frozen `BREAK_STAR` **and** `BOOK_STAR`. Formerly Part B of H-003.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Hypothesis**         | Measurement / entry windows materially change net Sharpe once the break gate and book rule are fixed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Economic rationale** | Estimation windows set how fast β, the z-score and the health gate react to regime change; the right speed depends on the architecture actually being traded.                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Test to complete**   | Sequential screens on fold-val (not fold-train Sharpe) → freeze window STARs; sealed OOS once.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Notes**              | **Runs after H-004 by design.** Fitting these windows on the frozen book and then handing them to both arms would bias the H-004 bake-off toward `freeze`, since the rotating arm would compete using windows tuned for its opponent. Grids unchanged (day units; ×6 on 1h): OLS/ADF `{504,252,126,63}`; entry `{1.5,2.0,2.5}`; z `{40,60,90}`. STARs: `OLS_WINDOW_STAR`, `ADF_WINDOW_STAR`, `ENTRY_Z_STAR`, `Z_WINDOW_STAR`. If `BOOK_STAR = rotate`, the discovery lookback `L` (pre-registered at 252 for H-004) may be screened **here** and nowhere earlier. Notebook: `H-005_window_screens.ipynb`. |


---



## H-006 · Coint · Kalman β vs static OLS hedge · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                     |
| **What it is**         | Time-varying hedge ratio via Kalman filter vs static OLS β; spread built point-in-time per top Notes. Requires frozen `BREAK_STAR` (H-003), `BOOK_STAR` (H-004) and the H-005 window STARs.                                                                                                                                                                         |
| **Hypothesis**         | Kalman β beats static OLS hedge on OOS spread Sharpe / max DD / corr to S1.                                                                                                                                                                                                                                                                                         |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                                                                   |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                   |
| **Test to complete**   | Traditional z-score entry/exit for both arms under frozen break + windows. Score candidates on fold-val (not train Sharpe). WF boxplots of Sharpe, max DD, corr to S1; sealed OOS once.                                                                                                                                                                             |
| **Notes**              | Entry/exit = traditional z-score (entry z / z-window from H-005 STARs). Spread history obeys PIT β Notes. **Freeze on fold-val Sharpe** (max DD / corr to S1 secondary); ADF / HL / lag-1 autocorr of `spread` are diagnostics only. **Impl:** 2-state KF tracks [\beta_t, \alpha_t] as a joint random walk; spread / returned β,α are from the **prior** \theta_{t |




Look to expand this test to see if a slower Kalman δ (not Chan 1e-4) improves tradable spread quality vs OLS; ADF/HL remain diagnostics.

---



## H-007 · Coint · ADX and/or RSI trend filter · 2026-08-10


| Field                  |                                                                                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                            |
| **What it is**         | Require ADX and/or RSI confirmation (or veto) before opening a spread trade.                                                                               |
| **Hypothesis**         | ADX and/or RSI confirmation/veto before entry improves net Sharpe / max DD / corr to S1 vs ungated entries.                                                |
| **Economic rationale** | —                                                                                                                                                          |
| **Data required**      | —                                                                                                                                                          |
| **Test to complete**   | Validation protocol: fold-val boxplots of Sharpe, max DD, corr to S1; sealed OOS once.                                                                     |
| **Notes**              | Prefer defaults (e.g. RSI 14, ADX 14) + at most a tiny pre-registered robustness set. Select via Validation protocol (fold-val boxplots; sealed OOS once). |


---



## H-008 · Coint · Half-life gate · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **What it is**         | Only trade pairs whose estimated mean-reversion half-life lies in a pre-registered band [L_{\min}, L_{\max}]. **First point in the stack where half-life is used at all.**                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Hypothesis**         | Half-life gate (only trade if half-life ∈ [Lmin, Lmax]) improves Sharpe / max DD / corr to S1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Economic rationale** | A spread whose half-life exceeds its own z-window cannot revert inside the window used to standardise it, so the entry is mistimed by construction — universe B failed exactly this way (best pair p=0.0501 with HL > 60 vs `Z_WINDOW=60`).                                                                                                                                                                                                                                                                                                                                                             |
| **Data required**      | Panel rolling `half_life`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Test to complete**   | Same traditional z-score entry as H-006. Primary band vs at most one alternate via WF val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Notes**              | Same traditional z-score entry as H-006; adds half-life clipping only. **Pre-register one** [L_{\min}, L_{\max}] from literature or economics as the primary band; allow **at most one** alternate band for robustness — **not** a grid over many bands. Choose between primary vs alternate (if any) via WF val boxplots; sealed OOS once. **Impl:** discrete half-life `-ln(2)/ln(1+b)` from `Δs_t = a + b·s_{t-1}`; NaN if `b >= 0` (no MR) or `1+b <= 0` (oscillatory). Store rolling series: `compute_half_life`; scalar discovery helper: `ou_half_life`. Notebook: `H-008_half_life_gate.ipynb`. |


**Half-life is deliberately absent before this hypothesis.** H-001 and H-004 rank candidates on **p-value only**; eligibility is EG pass plus minimum mutual bars. Using a tuned band earlier would choose a knob before the hypothesis that tests it. Accepted consequence: some selected pairs may have HL > `Z_WINDOW` and be effectively untradable, which is precisely what this hypothesis measures.

**Two jobs once frozen.**

1. **Universal trading gate** — in **both** arms, and regardless of how many candidates a pool holds, no pair with HL outside the band is traded.
2. **Selection filter in dynamic mode** — when `BOOK_STAR = rotate` and a pool has more EG-significant candidates than the per-pool cap of 2, the band filters and prioritises among them *before* p-value ranking decides.

---



## H-009 · Coint · Overlapping legs: allow vs never-allow · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                 |
| **What it is**         | Bake-off **allow overlapping legs** vs **never allow overlapping legs** (shared ticker across pairs).                                                                                                                                                                                                                                                                                           |
| **Hypothesis**         | Forbidding overlapping legs improves portfolio Sharpe / max DD / corr to S1 vs allowing overlaps, after costs.                                                                                                                                                                                                                                                                                  |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                               |
| **Test to complete**   | Allow vs never-allow under the Validation protocol (Sharpe, max DD, corr to S1).                                                                                                                                                                                                                                                                                                                |
| **Notes**              | **Never-allow rules:** Conflict = candidate shares **any ticker** with an already open pair, or with another candidate on the same bar. If a position is **already open** and a new signal shares a leg → **do not open** the new position. If **two** (or more) signals would open on the **same bar** and their legs overlap → open only the candidate with the best **Score × confidence** ( |


---



## H-010 · Coint · Exit: n half-lives + ATR SL + max-loss breaker · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **What it is**         | Time-stop after n half-lives; ATR-on-spread stop sized to fixed 1% portfolio loss; per-pair max-loss circuit breaker.                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Hypothesis**         | Time-stop after n half-lives, ATR-on-spread stop sized to fixed 1% loss, plus a per-pair max-loss circuit breaker, improves Sharpe / max DD / corr to S1 vs mean-exit-only.                                                                                                                                                                                                                                                                                                                                                             |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Test to complete**   | Compare to mean-exit-only under the Validation protocol (Sharpe, max DD, corr to S1). Discrete knobs (e.g. ATR multiple) scored on fold-val, not fold-train Sharpe.                                                                                                                                                                                                                                                                                                                                                                     |
| **Notes**              | **Time exit:** flat if not reverted within n \times half-life. **ATR stop:** position size so stop hit ≈ **1%** portfolio loss. **Max-loss circuit breaker:** if a **single pair’s** open PnL reaches a hard floor (example **−20%** — exact equity attribution fixed at implement), **liquidate that pair immediately** and do not re-enter until a reset rule. Catastrophe backstop separate from the 1% ATR risk unit. **Backtest:** path-check high/low after open entry (S1-style). **Live/paper:** resting stop/limit after fill. |


---



## H-011 · Coint · Kalman spread-correlation gate · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                            |
| **What it is**         | Block new pairs when Kalman-filtered correlation of spreads exceeds a threshold k; same conflict resolution as H-009.                                                                                                                                                                                                      |
| **Hypothesis**         | Blocking new pairs when                                                                                                                                                                                                                                                                                                    |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                          |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                          |
| **Test to complete**   | Bake-off no corr gate vs gate on; pre-register a small k set; choose on fold-val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                                                                                                                                                                   |
| **Notes**              | Estimate **time-varying correlation** between pair spread series (prefer spread **returns**/changes) with a **Kalman filter** tracking \hat\rho_t, **not** a fixed-window OLS/sample correlation. **Must use the single shared Kalman module** (state = correlation or bivariate moments → \hat\rho_t); no new KF core. If |


---



## H-012 · Coint · Score × confidence sizing · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                        |
| **What it is**         | Position size proportional to                                                                                                                                                                                                                                          |
| **Hypothesis**         | Sizing \propto                                                                                                                                                                                                                                                         |
| **Economic rationale** | —                                                                                                                                                                                                                                                                      |
| **Data required**      | —                                                                                                                                                                                                                                                                      |
| **Test to complete**   | Arms: equal risk vs score-only vs score×(1−p). Validation protocol boxplots; sealed OOS once.                                                                                                                                                                          |
| **Notes**              | ADF p = stationarity/coint evidence; (1-p) up-weights when stronger. Same Score × confidence formula is the **tie-break / priority metric** in H-009 and H-011 conflict rules; this hyp tests it as a **position-size** multiplier. Selection via Validation protocol. |


---



## H-013 · Coint · Vol-aware entry k_t vs S1 vol targeting · 2026-08-10


| Field                  |                                                                                                                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                         |
| **What it is**         | Vol-aware entry threshold k_t = k_0 \cdot \sigma_t / \bar\sigma (need a larger move in high-vol regimes) vs fixed k, compared to S1-style portfolio vol targeting.                      |
| **Hypothesis**         | k_t = k_0 \cdot \sigma_t / \bar\sigma beats fixed k on Sharpe / max DD / corr to S1; compare in-test to S1 portfolio vol targeting (S1 H-014 / `06_risk/s1_equities/vol_targeting.py`). |
| **Economic rationale** | —                                                                                                                                                                                       |
| **Data required**      | —                                                                                                                                                                                       |
| **Test to complete**   | Arms: fixed k vs vol-aware entry k_t vs S1-style portfolio VT. Validation protocol; sealed OOS once.                                                                                    |
| **Notes**              | Selection via Validation protocol. Entry-threshold lever vs position-leverage lever.                                                                                                    |


**Formulae**

- Vol-aware entry: k_t = k_0 \cdot \sigma_t / \bar\sigma
- S1-style vol targeting: L_t \propto \sigma_{\text{target}} / \hat\sigma_{\text{portfolio}}



## H-014 · Coint · Adaptive z-window (trad z) · 2026-08-12


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                   |
| **What it is**         | Bake-off fixed traditional z window vs a PIT adaptive standardization window driven by lagged rolling half-life.                                                                                                                                                                                                                                                                  |
| **Hypothesis**         | Adaptive `z_window_t = clip(2 * half_life_{t-1}, z_min, z_max)` improves OOS Sharpe / max DD / corr to S1 vs fixed `Z_WINDOW` from H-003 (`Z_WINDOW_STAR`, default 60).                                                                                                                                                                                                           |
| **Economic rationale** | Mean-reversion speed (and thus a sensible lookback for z) may change with regime; a fixed window cannot track that.                                                                                                                                                                                                                                                               |
| **Data required**      | Panel rolling `half_life`; traditional z entry/exit.                                                                                                                                                                                                                                                                                                                              |
| **Test to complete**   | Arms: fixed (H-003 `Z_WINDOW_STAR`) vs one pre-registered adaptive clamp band (at most one alternate band). Validation protocol fold-val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                                                                                                                                                                  |
| **Notes**              | Keeps traditional z definition; only the standardization window adapts. Do **not** conflate with H-008 half-life gate (trade / no-trade). **Pinch of salt:** fixed length was already screened under `BREAK_STAR` in H-005 (`Z_WINDOW_STAR` ∈ {40,60,90}) — adaptive bake-off may be overfit relative to that look. Lag HL one bar; clamp to avoid NaN/explosion when MR is weak. |


---



## H-015 · Coint · Entry extremity scores · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **What it is**         | Bake-off of separable alternatives to fixed traditional z for deciding when the spread is overbought/oversold. Implement only chosen subsets.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Hypothesis**         | Entry scores other than fixed traditional z improve OOS Sharpe / max DD / corr to S1 after costs (variants tested separately).                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Test to complete**   | Bake-off chosen subsets under the Validation protocol; same pairs/costs/exits where possible. Metrics: Sharpe, max DD, corr to S1.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Notes**              | Selectable variants (not one forced stack): (1) Rolling / EWM z + asymmetric bands — enter at \pm k_{\text{in}}, exit nearer mean or at k_{\text{out}} < k_{\text{in}}; rolling vs EWM vol. (2) OU / AR(1) residual score — see below. (3) Fused Kalman β + HMM regime + Kalman-on-spread innovation — β from shared Kalman; HMM gates mean-reverting vs trending/broken (flat when not MR); in MR only, trade standardized Kalman innovation on the spread (**reuse shared Kalman module** for β and spread state; no second KF core). (4) Copula / conditional quantile — see below. |


**OU / AR(1) residual score (A-level Maths)**

- A **pair spread** is one series after hedging (e.g. s_t = y_t - \beta x_t).
- Fit **AR(1)** on that spread: today’s spread ≈ pull toward a mean + leftover. **OU** = continuous-time twin.
- **Residual** = leftover / distance of s_t from \mu — from **spread on its own past**, not stock-on-stock (β already done).
- **Score:** leftover size vs model noise → overbought/oversold under mean reversion.

Steps: (1) build PIT spread (2) estimate \mu, \phi, \sigma on past only (3) distance of s_t from \mu in units of \sigma (4) enter beyond threshold.

**Copula / conditional quantile**

- **Leg** = one asset in the pair (leg A, leg B). Spread = weighted difference of the two legs.
- **Copula** = model of how the two legs move together (dependence) given each leg’s own distribution.
- **Conditional quantile:** given leg B today, is leg A extreme vs what the copula expects? That can fire without a large linear-spread z.

**Architecture (variant 3)**

```text
Prices → Kalman_β → Spread_s → HMM_regime
                              ├─ MR → Kalman_on_spread → std innovation entry
                              └─ trend/break → flat
```

---

