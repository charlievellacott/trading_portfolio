# Hypothesis Log

## Validation protocol

Mirror `03_models/s1_equities/model_tests/06_training_gbm_lambdarank_wf.ipynb`: purged expanding walk-forward on **research IS** → freeze by selection score + **boxplot review across folds** → **one** sealed OOS look for tearsheet / final backtest. Embargo between fold-train and fold-val (pairs analogue of the GBM 1-week embargo).

- **Fold-train (pairs):** estimate / warm anything that must be fit from past data inside the strategy (pair formation, PIT β / Kalman state, half-life estimate, OU/ADF inputs, corr KF state). Do **not** pick discrete design knobs by maximising train Sharpe.
- **Fold-val:** score each **pre-registered** candidate config (including knobs like ATR multiple) on that fold’s validation segment. Aggregate across folds (boxplots of Sharpe, max DD, corr to S1). Freeze the winner on research IS only; never tune on sealed OOS.
- **Research IS / sealed OOS:** research IS is the sleeve train panel through that universe’s fixed `RESEARCH_IS_END` (A `2020-12-31`, B `2022-12-31`, C `2021-12-31`); sealed OOS is after it. Pair list is frozen at discovery on `date <= T`. **If H-002 freezes 1H**, later hyps stay on the Yahoo 1H overlap window with a **70/30 IS:OOS** split (not the full C 1D census).
- **Stacking:** freeze one knob at a time **H-002 → H-013**. Never re-open an earlier STAR. You **type** the STAR after fold-val boxplots; notebooks must not assign STAR from `argmax` / median Sharpe (hint print only). JSON write and sealed OOS are gated on a non-None STAR via `require_star`.
- **Research stack:** `04_backtest/s2_coint` (WF / runner / report) + `05_strategies/s2_coint` engine/config. **Do not** add `s2_strategy.py` in this program — that file is a future hardcoded live recipe after all hyps freeze.
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

Do **not** add an extra `.shift(1)` on close-`t` features. Do **not** materialize next-bar open on the signal row as a feature. Same lag for **1D / 1H** (H-002). Universe C has **no 4H** arm.

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

**Traditional z baseline:** H-003 and H-006 use fixed-k rolling z-score entry/exit (`Z_WINDOW=60` on the panel). Adaptive z-window is H-012. Extremity-score alternatives are H-013.

**Shared conflict-resolution rules** (H-007 never-allow arm; H-009 when |\hat\rho_t| > k):

1. If a position is **already open** and a new candidate **conflicts** with it → **do not open** the new position.
2. If **both** candidates would open on the **same bar** → open only the one with the best **Score × confidence** (|\text{score}| \times (1 - p_{\text{ADF}}); score from the active entry rule / chosen H-013 variant when in use).

---


| ID    | Date       | Asset | Factor                                         | Data required | Status          |
| ----- | ---------- | ----- | ---------------------------------------------- | ------------- | --------------- |
| H-001 | 2026-08-10 | Coint | Universes A / B / C                            | —             | DECIDED         |
| H-002 | 2026-08-10 | Coint | 1D vs 1H after costs (no 4H for C)             | —             | NOT IMPLEMENTED |
| H-003 | 2026-08-10 | Coint | Kalman β vs static OLS hedge (trad z)          | —             | NOT IMPLEMENTED |
| H-004 | 2026-08-10 | Coint | Cointegration-break flat rule                  | —             | NOT IMPLEMENTED |
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
| **Status**             | DECIDED                                                                                                                            |
| **What it is**         | Compare candidate pair universes A, B, and C under the same trading rules.                                                         |
| **Hypothesis**         | Evaluate universes A, B, and C on Sharpe (after costs), max DD, and correlation to S1.                                             |
| **Economic rationale** | —                                                                                                                                  |
| **Data required**      | —                                                                                                                                  |
| **Test to complete**   | Validation protocol: WF fold-val boxplots of Sharpe, max DD, corr to S1; freeze universe on research IS; one sealed OOS tearsheet. |
| **Notes**              | **Frozen:** universe **C refined**. Locked pairs: `1398.HK\|0939.HK`, `1288.HK\|3328.HK`, `8306.T\|8316.T`. `RESEARCH_IS_END=2021-12-31`. A/B failed; do **not** re-pick or retrofit H-001 with walk-forward. H-001 keep-gate empty list locks **zero** pairs (unlike the panel notebook, which keeps all EG passers if the keep list is empty) — later hyps must use the three locked IDs in `04_backtest/s2_coint/artifacts/s2_star_stack.json`. IS trad-z book being net-negative is **not** a reason to reopen the universe. Notebook: `H-001_universes.ipynb`. |


---



## H-002 · Coint · 1D vs 1H after costs · 2026-08-10


| Field                  |                                                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                         |
| **What it is**         | Same pair logic on daily vs 1H bars, with timeframe-appropriate frictions. **No 4H for universe C.**                    |
| **Hypothesis**         | 1D timeframe survives costs better than 1H after frictions.                                                             |
| **Economic rationale** | —                                                                                                                       |
| **Data required**      | Yahoo 1D (existing C panel) and Yahoo 1H (~730d overlap). Locked C pairs only.                                          |
| **Test to complete**   | Compare on Sharpe, max DD, corr to S1 under the Validation protocol (identical pair set and entry rule).                |
| **Notes**              | **No 4H anywhere in this sleeve for C.** Same Timing contract at 1D / 1H (close `t` → fill open `t+1`). Per-fill `COSTS` (not scaled down on 1H). Sharpe annualizes from bar count (`periods_per_year_from_index`). Scale OLS/z/HL lookbacks to **sessions** (`252/60/252` days × 6 1H bars/session), not raw hour counts. Sample: longest overlapping Yahoo 1H window; WF on first 70% of that overlap; one sealed look on the last 30%. **If 1D wins**, later hyps use full C 1D with `RESEARCH_IS_END=2021-12-31`. **If 1H wins**, stay on that window with 70/30 IS:OOS. Type `BAR_STAR` after boxplots. Notebook: `H-002_bar_size.ipynb`. |


---



## H-003 · Coint · Kalman β vs static OLS hedge · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **What it is**         | Time-varying hedge ratio via Kalman filter vs static OLS β; spread built point-in-time per top Notes.                                                                                                                                                                                                                                                                                                                                                                        |
| **Hypothesis**         | Kalman β beats static OLS hedge on OOS spread Sharpe / max DD / corr to S1.                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Test to complete**   | Traditional z-score entry/exit for both arms. Score candidates on fold-val (not train Sharpe). WF boxplots of Sharpe, max DD, corr to S1; sealed OOS once.                                                                                                                                                                                                                                                                                                                   |
| **Notes**              | Entry/exit = traditional z-score. Spread history obeys PIT β Notes. Isolates hedge quality before gates or alternate scores. Score candidates on fold-val (not train Sharpe). **Impl:** 2-state KF tracks [\beta_t, \alpha_t] as a joint random walk; spread / returned β,α are from the **prior** \theta_{t\|t-1} (innovation), never the posterior. Shared core: `feature_implementation/kalman.py`. Store: `compute_kalman_hedge_spread` vs `compute_static_hedge_spread`. **Arms:** OLS 252 vs Kalman (`delta=1e-4`, `obs_var=1e-3`, `burn_in=30` days, session-scaled on 1H). Extra rolling ADF/HL diagnostic is **not** a freeze input. Type `HEDGE_STAR`. Notebook: `H-003_hedge.ipynb`. |


Look to expand this test to see if a kalman filter improves rolling ADF & half life

---



## H-004 · Coint · Cointegration-break flat rule · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                   |
| **What it is**         | Go flat when a cointegration / stationarity break is detected on the live spread.                                                                                                                                                                                 |
| **Hypothesis**         | Cointegration-break flat rule improves max DD (and Sharpe / corr to S1) more than it hurts return.                                                                                                                                                                |
| **Economic rationale** | —                                                                                                                                                                                                                                                                 |
| **Data required**      | —                                                                                                                                                                                                                                                                 |
| **Test to complete**   | Validation protocol: Sharpe, max DD, corr to S1 on fold-val boxplots; sealed OOS once.                                                                                                                                                                            |
| **Notes**              | **Impl health metrics** via `compute_coint_metrics`: `adf_pvalue` (rolling ADF on PIT spread) and `variance_jump` (recent spread std / lagged baseline std). Use as kill-switch / entry gate; discovery EG (`run_cointegration_test`) stays out of the live loop. **Arms:** `off` \| `block_05_flat_10` \| `flat_05`. Type `BREAK_STAR`. Notebook: `H-004_coint_break.ipynb`. |


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
| **Notes**              | RSI-14 / ADX-14 on the **spread** (constructed H/L envelope from close-t α/β), not the legs. **Arms:** `off` \| `adx_veto` (ADX>25) \| `rsi_confirm` (long RSI<30, short RSI>70) \| `both`. Type `TREND_STAR`. Notebook: `H-005_trend_filter.ipynb`. |


---



## H-006 · Coint · Half-life gate · 2026-08-10


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **What it is**         | Only trade pairs whose estimated mean-reversion half-life lies in a pre-registered band [L_{\min}, L_{\max}].                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Hypothesis**         | Half-life gate (only trade if half-life ∈ [Lmin, Lmax]) improves Sharpe / max DD / corr to S1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Economic rationale** | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Test to complete**   | Same traditional z-score entry as H-003. Primary band vs at most one alternate via WF val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Notes**              | Same traditional z-score entry as H-003; adds half-life clipping only. **Arms:** `off` \| `[5,60]` \| `[5,30]`. **Impl:** discrete half-life `-ln(2)/ln(1+b)` from `Δs_t = a + b·s_{t-1}`; NaN if `b >= 0` (no MR) or `1+b <= 0` (oscillatory). Store rolling series: `compute_half_life`; scalar discovery helper: `ou_half_life`. Type `HL_GATE_STAR`. Notebook: `H-006_half_life_gate.ipynb`. |


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
| **Notes**              | **Never-allow rules:** Conflict = candidate shares **any ticker** with an already open pair, or with another candidate on the same bar. If a position is **already open** and a new signal shares a leg → **do not open** the new position. If **two** (or more) signals would open on the **same bar** and their legs overlap → open only the candidate with the best **Score × confidence** (`|score|*(1-adf_p)`). **Arms:** `allow` vs `never_allow`. Pair list is never re-screened. Type `OVERLAP_STAR`. Notebook: `H-007_overlap.ipynb`. |


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
| **Notes**              | **Time exit:** flat if not reverted within n=3 × half-life. **ATR stop:** dollar risk at stop = `0.01 * pair_scale * L_t` of a scale=1 trade (do **not** shrink the stop distance to cancel scale). Wilder-14 ATR on constructed spread H/L. **Max-loss circuit breaker:** −20% pair; re-enter when ADF<0.05 and HL in frozen band or `[5,60]`. **Backtest:** path-check high/low after open entry. **Arms:** `mean_only` vs `hl3_atr_breaker`. Type `EXIT_STAR`. Notebook: `H-008_exits.ipynb`. |


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
| **Notes**              | Estimate **time-varying correlation** between pair spread **changes** with the shared Kalman module (`kalman_correlation`); no new KF core. Same conflict resolution as H-007. **Arms:** `off` \| `k=0.50` \| `k=0.70`. Type `CORR_GATE_STAR`. Notebook: `H-009_corr_gate.ipynb`. |


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
| **Notes**              | ADF p = stationarity/coint evidence; (1-p) up-weights when stronger. Same Score × confidence formula is the **tie-break / priority metric** in H-007 and H-009 conflict rules; this hyp tests it as a **position-size** multiplier. **Arms:** `equal` \| `score` \| `score_conf`. Fold-train mean abs score rescales to ~1. Type `SIZE_STAR`. Notebook: `H-010_sizing.ipynb`. |


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
| **Notes**              | Entry-threshold lever vs position-leverage lever. **Arms:** `fixed_k` \| `kt` (`k_t=2*σ_t/σ_bar`) \| `s1_vt` via `risk.s1_equities.vol_targeting`. Type `VOL_STAR`. Notebook: `H-011_vol.ipynb`. |


**Formulae**

- Vol-aware entry: k_t = k_0 \cdot \sigma_t / \bar\sigma
- S1-style vol targeting: L_t \propto \sigma_{\text{target}} / \hat\sigma_{\text{portfolio}}
## H-012 · Coint · Adaptive z-window (trad z) · 2026-08-12


| Field                  |                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                                                                                                                                                                          |
| **What it is**         | Bake-off fixed traditional z window vs a PIT adaptive standardization window driven by lagged rolling half-life.                                                                                                                                                                                                                                         |
| **Hypothesis**         | Adaptive `z_window_t = clip(2 * half_life_{t-1}, z_min, z_max)` improves OOS Sharpe / max DD / corr to S1 vs fixed `Z_WINDOW=60`.                                                                                                                                                                                                                         |
| **Economic rationale** | Mean-reversion speed (and thus a sensible lookback for z) may change with regime; a fixed window cannot track that.                                                                                                                                                                                                                                      |
| **Data required**      | Panel rolling `half_life`; traditional z entry/exit.                                                                                                                                                                                                                                                                                                     |
| **Test to complete**   | Arms: fixed 60 vs one pre-registered adaptive clamp band (at most one alternate band). Validation protocol fold-val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                                                                                                                                                              |
| **Notes**              | Keeps traditional z definition; only the standardization window adapts. Do **not** conflate with H-006 half-life gate (trade / no-trade). Panel v1 ships fixed `Z_WINDOW=60` only — this hyp is the adaptive arm. Lag HL one bar; clamp to avoid NaN/explosion when MR is weak. **Arms:** fixed 60 vs `clip(2*HL_{t-1}, 20, 120)` vs alt `[10,252]`. Type `Z_WINDOW_MODE_STAR`. Notebook: `H-012_z_window.ipynb`. |


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
| **Notes**              | Selectable variants (not one forced stack): (1) Rolling / EWM z + asymmetric bands — enter at \pm k_{\text{in}}, exit nearer mean or at k_{\text{out}} < k_{\text{in}}; rolling vs EWM vol. (2) OU / AR(1) residual score — see below. (3) Fused Kalman β + HMM regime + Kalman-on-spread innovation — β from shared Kalman; HMM gates mean-reverting vs trending/broken (flat when not MR); in MR only, trade standardized Kalman innovation on the spread (**reuse shared Kalman module** for β and spread state; no second KF core). **This program tests variants 1–3 only. No copula.** Type `ENTRY_STAR`. Notebook: `H-013_entry_scores.ipynb`. |


**OU / AR(1) residual score (A-level Maths)**

- A **pair spread** is one series after hedging (e.g. s_t = y_t - \beta x_t).
- Fit **AR(1)** on that spread: today’s spread ≈ pull toward a mean + leftover. **OU** = continuous-time twin.
- **Residual** = leftover / distance of s_t from \mu — from **spread on its own past**, not stock-on-stock (β already done).
- **Score:** leftover size vs model noise → overbought/oversold under mean reversion.

Steps: (1) build PIT spread (2) estimate \mu, \phi, \sigma on past only (3) distance of s_t from \mu in units of \sigma (4) enter beyond threshold.

**Copula / conditional quantile (deferred)**

Not in this program (H-013 variants 1–3 only). Kept here as background: copula = dependence of two legs given each marginal; conditional quantile asks whether leg A is extreme given leg B without a large linear-spread z.

**Architecture (variant 3)**

```text
Prices → Kalman_β → Spread_s → HMM_regime
                              ├─ MR → Kalman_on_spread → std innovation entry
                              └─ trend/break → flat
```

---



