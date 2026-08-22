# Hypothesis Log

## Validation protocol

Mirror `03_models/s1_equities/model_tests/06_training_gbm_lambdarank_wf.ipynb`: purged expanding walk-forward on **research IS** → freeze by selection score + **boxplot review across folds** → **one** sealed OOS look for tearsheet / final backtest. Embargo between fold-train and fold-val (pairs analogue of the GBM 1-week embargo).

- **Fold-train (pairs):** estimate / warm anything that must be fit from past data inside the strategy (pair formation, PIT β / Kalman state, half-life estimate, OU/ADF inputs, corr KF state). Do **not** pick discrete design knobs by maximising train Sharpe.
- **Fold-val:** score each **pre-registered** candidate config (including knobs like ATR multiple) on that fold’s validation segment. Aggregate across folds (boxplots of Sharpe, max DD, corr to S1). Freeze the winner on research IS only; never tune on sealed OOS.
- **Research IS / sealed OOS:** research IS is the sleeve train panel through that universe’s fixed `RESEARCH_IS_END` (A `2020-12-31`, B `2022-12-31`, C `2021-12-31`); sealed OOS is after it. Pair list is frozen at discovery on `date <= T`.
- **Primary metrics (every hyp):** net **Sharpe**, **max drawdown**, and **correlation to S1**.

### Timing contract (all S2 hyps / all bar sizes)

Not Default same-bar close-fill; not S1 trade-date / `feature_date` indexing.

| Role | Timestamp |
|------|-----------|
| Features / signal / decision | Close of bar `t` |
| Optional alt data | After close `t`, before open `t+1` (if knowable then) |
| Orders | Queued just before open `t+1` |
| Fill | Open of `t+1` (both legs; no-auction venues: next-bar OHLCV open) |
| First PnL / HL stops | From open `t+1` onward |

Do **not** add an extra `.shift(1)` on close-`t` features. Do **not** materialize next-bar open on the signal row as a feature. Same lag for 1D / 4H / 1H (H-002).

---



## Notes

> **Spread history vs β (must confirm):** When building a rolling spread series for z-scores, half-life, ADF, etc., does s_{t-k} use **β at that past time** \beta_{t-k} (point-in-time: s_{t-k} = y_{t-k} - \beta_{t-k} x_{t-k} - \alpha_{t-k}), or does it reuse the **current** β \beta_t on past prices (look-ahead / revision of history)?
>
> **Default for this sleeve:** use **β (and α) as of each timestamp** — never rewrite past spreads with today’s β. Kalman paths emit \beta_t each bar; static OLS must use only information available at that bar (rolling/expanding window ending at t-k), not a full-sample β.


### Implementation conventions (math / store)

- **Public API:** `01_data/processing/s2_coint_store.py` (`compute_*` / `run_cointegration_test` / `build_pair_panel`). Math lives in `feature_implementation/cointegration.py`; shared KF core in `feature_implementation/kalman.py`. Universe candidacy: `01_data/processing/s2_universe.py`.
- **Panel schema:** row `date = t` is the signal bar. Columns `open_y/high_y/low_y/close_y` and `open_x/high_x/low_x/close_x` (no `price_y` / `price_x`). Hedge, z, ADF, half-life use **`close_y` / `close_x` only**. Open/high/low are for fill/stop/PnL path — never inputs to β. Fill for a signal on `t` = next pair-bar `open_*` (lookup at backtest time; do not store `shift(-1)` open on the signal row).
- **Log prices:** store converts raw closes via `to_log_price`; math expects logs. Spread: \(s_t = y_t - \alpha_t - \beta_t x_t\) on closes.
- **Engle-Granger discovery:** test both `y~x` and `x~y`, keep lower p-value; separate OLS for α/β. `COINT_PVALUE = 0.05` flat — EG has low finite-sample power on small pre-specified lists; holdout confirmation is the real gate (no Bonferroni on a handful of economic pairs).
- **Bayesian / MCMC β:** deferred. Full-sample MCMC / smoother paths rewrite history with future info and violate PIT; if revisited, calibrate noise on fold-train only, then run the recursive filter live.

**Traditional z baseline:** H-004 and H-006 use fixed-k rolling z-score entry/exit (default `Z_WINDOW=60` unless `Z_WINDOW_STAR` frozen in H-003). Adaptive z-window is H-012. Extremity-score alternatives are H-013.

**Shared conflict-resolution rules** (H-007 never-allow arm; H-009 when |\hat\rho_t| > k):

1. If a position is **already open** and a new candidate **conflicts** with it → **do not open** the new position.
2. If **both** candidates would open on the **same bar** → open only the one with the best **Score × confidence** (|\text{score}| \times (1 - p_{\text{ADF}}); score from the active entry rule / chosen H-013 variant when in use).

---


| ID    | Date       | Asset | Factor                                         | Data required | Status          |
| ----- | ---------- | ----- | ---------------------------------------------- | ------------- | --------------- |
| H-001 | 2026-08-10 | Coint | Universes A / B / C                            | —             | DECIDED (Universe C) |
| H-002 | 2026-08-10 | Coint | 1D vs 4H / 1H after costs                      | —             | NOT IMPLEMENTED |
| H-003 | 2026-08-10 | Coint | Cointegration-break flat rule + window screens | —             | NOT IMPLEMENTED |
| H-004 | 2026-08-10 | Coint | Kalman β vs static OLS hedge (trad z)          | —             | NOT IMPLEMENTED |
| H-005 | 2026-08-10 | Coint | ADX and/or RSI trend filter                    | —             | NOT IMPLEMENTED |
| H-006 | 2026-08-10 | Coint | Half-life gate (trad z)                        | —             | NOT IMPLEMENTED |
| H-007 | 2026-08-10 | Coint | Overlapping legs: allow vs never-allow         | —             | NOT IMPLEMENTED |
| H-008 | 2026-08-10 | Coint | Exit: n half-lives + ATR SL + max-loss breaker | —             | NOT IMPLEMENTED |
| H-009 | 2026-08-10 | Coint | Kalman spread-correlation gate                 | —             | NOT IMPLEMENTED |
| H-010 | 2026-08-10 | Coint | Score × confidence sizing                      | —             | NOT IMPLEMENTED |
| H-011 | 2026-08-10 | Coint | Vol-aware entry k_t vs S1 vol targeting        | —             | NOT IMPLEMENTED |
| H-012 | 2026-08-12 | Coint | Adaptive z-window (trad z)                     | —             | NOT IMPLEMENTED |
| H-013 | 2026-08-10 | Coint | Entry extremity scores                         | —             | NOT IMPLEMENTED |

## H-001 · Coint · Universes A / B / C · 2026-08-10


| Field                  |                                                                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                    |
| **What it is**         | Compare candidate pair universes A, B, and C under the same trading rules.                                                         |
| **Hypothesis**         | Evaluate universes A, B, and C on Sharpe (after costs), max DD, and correlation to S1.                                             |
| **Economic rationale** | —                                                                                                                                  |
| **Data required**      | —                                                                                                                                  |
| **Test to complete**   | Validation protocol: WF fold-val boxplots of Sharpe, max DD, corr to S1; freeze universe on research IS; one sealed OOS tearsheet. |
| **Notes**              | Lock chosen universe before later hyps; do not re-pick after downstream bake-offs. H-001 notebook is `02_research/s2_coint/notebooks/hypothesis_tests/H-001_universes.ipynb` and runs (i) sector-only screening, (ii) manual keep gate, and (iii) research-IS trad-z diagnostics. Baseline constants (`ENTRY_Z=2`, `EXIT_Z=0`, beta-sized legs) and market costs (OANDA FX, Kraken spot, IBKR HK, IBKR JP) live in `strategies.s2_coint`. Notebook scores **research IS only** — per-pair trade count, median hold, cost bps/year, Sharpe / max DD, and rolling ADF. Do not use sealed OOS to pick the universe. Asia C IS postmortem → `02_research/s2_coint/notebooks/other_tests/01_asia_c_failure_diagnosis.ipynb`. |


---



## H-002 · Coint · 1D vs 4H / 1H after costs · 2026-08-10


| Field                  |                                                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                         |
| **What it is**         | Same pair logic on daily vs 4H vs 1H bars, with timeframe-appropriate frictions.                                        |
| **Hypothesis**         | 1D timeframe survives costs better than 4H/1H after frictions.                                                          |
| **Economic rationale** | —                                                                                                                       |
| **Data required**      | —                                                                                                                       |
| **Test to complete**   | Compare on Sharpe, max DD, corr to S1 under the Validation protocol (identical pair set and entry rule where possible). |
| **Notes**              | Same Timing contract at **1D / 4H / 1H** (signal close `t` → fill open `t+1`). Compare on Sharpe, max DD, corr to S1 under the Validation protocol. **Bar only** — OLS / ADF / entry / z screens live in H-003 Part B under frozen `BREAK_STAR` (Pipeline A). Notebook: `H-002_bar_size.ipynb`. |


---



## H-003 · Coint · Cointegration-break flat rule + window screens · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                   |
| **What it is**         | Part A: go flat / block when cointegration / stationarity breaks on the live spread. Part B (only if Part A continues): sequential screens for OLS lookback, ADF window, entry z, and fixed z-window **under** frozen `BREAK_STAR`. |
| **Hypothesis**         | Cointegration-break flat rule improves max DD (and Sharpe / corr to S1) more than it hurts return; then measurement / entry knobs refine the gated book. |
| **Economic rationale** | —                                                                                                                                                                                                                                                                 |
| **Data required**      | —                                                                                                                                                                                                                                                                 |
| **Test to complete**   | Validation protocol: Part A break arms on fold-val boxplots → sleeve kill check → freeze `BREAK_STAR`; Part B sequential screens → freeze window STARs; sealed OOS once. |
| **Notes**              | **Impl health metrics** via `compute_coint_metrics`: `adf_pvalue` (rolling ADF on PIT spread) and `variance_jump`. Discovery EG stays out of the live loop. Soft defaults for Part A: ols/adf 252d, entry_z 2, z 60. Part B grids (day units; ×6 on 1h): OLS/ADF `{504,252,126,63}`; entry `{1.5,2.0,2.5}`; z `{40,60,90}`. STARs: `BREAK_STAR`, `OLS_WINDOW_STAR`, `ADF_WINDOW_STAR`, `ENTRY_Z_STAR`, `Z_WINDOW_STAR`. Short borrow not modeled. Notebook: `H-003_coint_break.ipynb`. |

**Sleeve kill / continue (Part A only — before Part B).** Metric = median fold-val `ann_sharpe` (net). Do not use sealed OOS.

| Band | Median fold-val Sharpe | Action |
|------|------------------------|--------|
| Severe | **≤ −0.50** | Shelve sleeve — no Part B / H-004+ |
| Deeply negative | **(−0.50, −0.25]** | Shelve unless beats `off` by **≥ +0.25** Sharpe **or** median max DD **≥ 5 pp** better **and** `pct_adf_lt_0.05` **≥ +10 pp** vs `off` |
| Soft negative | **(−0.25, 0]** | Yellow — continue only if break clearly helped; document why |
| Non-negative | **> 0** | Continue to Part B |


---



## H-004 · Coint · Kalman β vs static OLS hedge · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **What it is**         | Time-varying hedge ratio via Kalman filter vs static OLS β; spread built point-in-time per top Notes. Requires frozen `BREAK_STAR` and H-003 window STARs.                                                                                                                                                                                                                                                                                                                                                                        |
| **Hypothesis**         | Kalman β beats static OLS hedge on OOS spread Sharpe / max DD / corr to S1.                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Test to complete**   | Traditional z-score entry/exit for both arms under frozen break + windows. Score candidates on fold-val (not train Sharpe). WF boxplots of Sharpe, max DD, corr to S1; sealed OOS once.                                                                                                                                                                                                                                                                                                                   |
| **Notes**              | Entry/exit = traditional z-score (entry z / z-window from H-003 STARs). Spread history obeys PIT β Notes. **Freeze on fold-val Sharpe** (max DD / corr to S1 secondary); ADF / HL / lag-1 autocorr of `spread` are diagnostics only. **Impl:** 2-state KF tracks [\beta_t, \alpha_t] as a joint random walk; spread / returned β,α are from the **prior** \theta_{t\|t-1} (innovation), never the posterior. Shared core: `feature_implementation/kalman.py`. Store: `compute_kalman_hedge_spread` vs `compute_static_hedge_spread`. Chan (2013) \(\delta=10^{-4}\), \(R=10^{-3}\) makes \(Q/R\) large enough that innovations are nearly white — diagnostic-only, not a freeze candidate. Slower literature: QuantStart/O’Mahony \(\delta=10^{-5}\) with \(R=1\); Palomar §15.6 / Feng & Palomar (2016) \(\alpha=10^{-5}\) (basic) / \(10^{-6}\) (momentum). **Arms:** OLS vs Kalman \(\delta\in\{10^{-5},10^{-6},10^{-7}\}\) (`obs_var=1e-3`, `burn_in=30` days, session-scaled on 1H; \(\delta\) is **not** session-scaled on 1D — if 1H is re-opened, \(Q\approx Q_{\mathrm{day}}/6\)). Type `HEDGE_STAR` (`ols`\|`kalman`) and, if Kalman, `KALMAN_DELTA_STAR`. Notebook: `H-004_hedge.ipynb`. |


Look to expand this test to see if a slower Kalman δ (not Chan 1e-4) improves tradable spread quality vs OLS; ADF/HL remain diagnostics.

---



## H-005 · Coint · ADX and/or RSI trend filter · 2026-08-10


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



## H-006 · Coint · Half-life gate · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **What it is**         | Only trade pairs whose estimated mean-reversion half-life lies in a pre-registered band [L_{\min}, L_{\max}].                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Hypothesis**         | Half-life gate (only trade if half-life ∈ [Lmin, Lmax]) improves Sharpe / max DD / corr to S1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Test to complete**   | Same traditional z-score entry as H-004. Primary band vs at most one alternate via WF val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Notes**              | Same traditional z-score entry as H-004; adds half-life clipping only. **Pre-register one** [L_{\min}, L_{\max}] from literature or economics as the primary band; allow **at most one** alternate band for robustness — **not** a grid over many bands. Choose between primary vs alternate (if any) via WF val boxplots; sealed OOS once. **Impl:** discrete half-life `-ln(2)/ln(1+b)` from `Δs_t = a + b·s_{t-1}`; NaN if `b >= 0` (no MR) or `1+b <= 0` (oscillatory). Store rolling series: `compute_half_life`; scalar discovery helper: `ou_half_life`. |


---



## H-007 · Coint · Overlapping legs: allow vs never-allow · 2026-08-10


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



## H-008 · Coint · Exit: n half-lives + ATR SL + max-loss breaker · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                           |
| **What it is**         | Time-stop after n half-lives; ATR-on-spread stop sized to fixed 1% portfolio loss; per-pair max-loss circuit breaker.                                                                                                                                                                                                                                                                                                     |
| **Hypothesis**         | Time-stop after n half-lives, ATR-on-spread stop sized to fixed 1% loss, plus a per-pair max-loss circuit breaker, improves Sharpe / max DD / corr to S1 vs mean-exit-only.                                                                                                                                                                                                                                               |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Test to complete**   | Compare to mean-exit-only under the Validation protocol (Sharpe, max DD, corr to S1). Discrete knobs (e.g. ATR multiple) scored on fold-val, not fold-train Sharpe.                                                                                                                                                                                                                                                       |
| **Notes**              | **Time exit:** flat if not reverted within n \times half-life. **ATR stop:** position size so stop hit ≈ **1%** portfolio loss. **Max-loss circuit breaker:** if a **single pair’s** open PnL reaches a hard floor (example **−20%** — exact equity attribution fixed at implement), **liquidate that pair immediately** and do not re-enter until a reset rule. Catastrophe backstop separate from the 1% ATR risk unit. **Backtest:** path-check high/low after open entry (S1-style). **Live/paper:** resting stop/limit after fill. |


---



## H-009 · Coint · Kalman spread-correlation gate · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                            |
| **What it is**         | Block new pairs when Kalman-filtered correlation of spreads exceeds a threshold k; same conflict resolution as H-007.                                                                                                                                                                                                      |
| **Hypothesis**         | Blocking new pairs when                                                                                                                                                                                                                                                                                                    |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                          |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                          |
| **Test to complete**   | Bake-off no corr gate vs gate on; pre-register a small k set; choose on fold-val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                                                                                                                                                                   |
| **Notes**              | Estimate **time-varying correlation** between pair spread series (prefer spread **returns**/changes) with a **Kalman filter** tracking \hat\rho_t, **not** a fixed-window OLS/sample correlation. **Must use the single shared Kalman module** (state = correlation or bivariate moments → \hat\rho_t); no new KF core. If |


---



## H-010 · Coint · Score × confidence sizing · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                        |
| **What it is**         | Position size proportional to                                                                                                                                                                                                                                          |
| **Hypothesis**         | Sizing \propto                                                                                                                                                                                                                                                         |
| **Economic rationale** | —                                                                                                                                                                                                                                                                      |
| **Data required**      | —                                                                                                                                                                                                                                                                      |
| **Test to complete**   | Arms: equal risk vs score-only vs score×(1−p). Validation protocol boxplots; sealed OOS once.                                                                                                                                                                          |
| **Notes**              | ADF p = stationarity/coint evidence; (1-p) up-weights when stronger. Same Score × confidence formula is the **tie-break / priority metric** in H-007 and H-009 conflict rules; this hyp tests it as a **position-size** multiplier. Selection via Validation protocol. |


---



## H-011 · Coint · Vol-aware entry k_t vs S1 vol targeting · 2026-08-10


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
## H-012 · Coint · Adaptive z-window (trad z) · 2026-08-12


| Field                  |                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                          |
| **What it is**         | Bake-off fixed traditional z window vs a PIT adaptive standardization window driven by lagged rolling half-life.                                                                                                                                                                                                                                         |
| **Hypothesis**         | Adaptive `z_window_t = clip(2 * half_life_{t-1}, z_min, z_max)` improves OOS Sharpe / max DD / corr to S1 vs fixed `Z_WINDOW` from H-003 (`Z_WINDOW_STAR`, default 60).                                                                                                                                                                                                                         |
| **Economic rationale** | Mean-reversion speed (and thus a sensible lookback for z) may change with regime; a fixed window cannot track that.                                                                                                                                                                                                                                      |
| **Data required**      | Panel rolling `half_life`; traditional z entry/exit.                                                                                                                                                                                                                                                                                                     |
| **Test to complete**   | Arms: fixed (H-003 `Z_WINDOW_STAR`) vs one pre-registered adaptive clamp band (at most one alternate band). Validation protocol fold-val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                                                                                                                                                              |
| **Notes**              | Keeps traditional z definition; only the standardization window adapts. Do **not** conflate with H-006 half-life gate (trade / no-trade). **Pinch of salt:** fixed length was already screened under `BREAK_STAR` in H-003 Part B (`Z_WINDOW_STAR` ∈ {40,60,90}) — adaptive bake-off may be overfit relative to that look. Lag HL one bar; clamp to avoid NaN/explosion when MR is weak. |


---



## H-013 · Coint · Entry extremity scores · 2026-08-10


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



