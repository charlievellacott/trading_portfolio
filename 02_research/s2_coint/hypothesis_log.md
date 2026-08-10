# Hypothesis Log

## Validation protocol

Mirror `03_models/s1_equities/model_tests/06_training_gbm_lambdarank_wf.ipynb`: purged expanding walk-forward on **research IS** → freeze by selection score + **boxplot review across folds** → **one** sealed OOS look for tearsheet / final backtest. Embargo between fold-train and fold-val (pairs analogue of the GBM 1-week embargo).

- **Fold-train (pairs):** estimate / warm anything that must be fit from past data inside the strategy (pair formation, PIT β / Kalman state, half-life estimate, OU/ADF inputs, corr KF state). Do **not** pick discrete design knobs by maximising train Sharpe.
- **Fold-val:** score each **pre-registered** candidate config (including knobs like ATR multiple) on that fold’s validation segment. Aggregate across folds (boxplots of Sharpe, max DD, corr to S1). Freeze the winner on research IS only; never tune on sealed OOS.
- **Primary metrics (every hyp):** net **Sharpe**, **max drawdown**, and **correlation to S1**.

---

## Notes

> **Spread history vs β (must confirm):** When building a rolling spread series for z-scores, half-life, ADF, etc., does \(s_{t-k}\) use **β at that past time** \(\beta_{t-k}\) (point-in-time: \(s_{t-k} = y_{t-k} - \beta_{t-k} x_{t-k} - \alpha_{t-k}\)), or does it reuse the **current** β \(\beta_t\) on past prices (look-ahead / revision of history)?
>
> **Default for this sleeve:** use **β (and α) as of each timestamp** — never rewrite past spreads with today’s β. Kalman paths emit \(\beta_t\) each bar; static OLS must use only information available at that bar (rolling/expanding window ending at \(t-k\)), not a full-sample β.

> **Kalman reuse (mandatory for implementers):** There is **no** Kalman implementation in the repo yet (only an S1 “potential ideas” note). The **first** S2 feature that needs a Kalman filter must create **one** shared module (prefer under `01_data/processing/` beside peers, e.g. a single `kalman.py` or equivalent — exact path chosen at implement time) and expose reusable primitives (predict/update, optional state helpers).
>
> **Every** later Kalman use — hedge β (H-003), spread-state / innovation (H-005 variant 3), **time-varying spread correlation (H-010)**, or any other — **must import and call that module**. Do **not** create a second Kalman filter module, notebook-local KF, or copy-pasted filter. Thin wrappers for different state definitions (β, spread, correlation) are fine; a second filter core is not.

**Traditional z baseline:** H-003 and H-004 use fixed-\(k\) rolling z-score entry/exit. Extremity-score alternatives are H-005.

**Shared conflict-resolution rules** (H-009 never-allow arm; H-010 when \(|\hat\rho_t| > k\)):

1. If a position is **already open** and a new candidate **conflicts** with it → **do not open** the new position.
2. If **both** candidates would open on the **same bar** → open only the one with the best **Score × confidence** (\(|\text{score}| \times (1 - p_{\text{ADF}})\); score from the active entry rule / chosen H-005 variant when in use).

---


| ID    | Date       | Asset | Factor                                              | Data required | Status          |
| ----- | ---------- | ----- | --------------------------------------------------- | ------------- | --------------- |
| H-001 | 2026-08-10 | Coint | Universes A / B / C                                 | —             | NOT IMPLEMENTED |
| H-002 | 2026-08-10 | Coint | 1D vs 4H / 1H after costs                           | —             | NOT IMPLEMENTED |
| H-003 | 2026-08-10 | Coint | Kalman β vs static OLS hedge (trad z)               | —             | NOT IMPLEMENTED |
| H-004 | 2026-08-10 | Coint | Half-life gate (trad z)                             | —             | NOT IMPLEMENTED |
| H-005 | 2026-08-10 | Coint | Entry extremity scores                              | —             | NOT IMPLEMENTED |
| H-006 | 2026-08-10 | Coint | ADX and/or RSI trend filter                         | —             | NOT IMPLEMENTED |
| H-007 | 2026-08-10 | Coint | Exit: n half-lives + ATR SL + max-loss breaker      | —             | NOT IMPLEMENTED |
| H-008 | 2026-08-10 | Coint | Cointegration-break flat rule                       | —             | NOT IMPLEMENTED |
| H-009 | 2026-08-10 | Coint | Overlapping legs: allow vs never-allow              | —             | NOT IMPLEMENTED |
| H-010 | 2026-08-10 | Coint | Kalman spread-correlation gate                      | —             | NOT IMPLEMENTED |
| H-011 | 2026-08-10 | Coint | Score × confidence sizing                           | —             | NOT IMPLEMENTED |
| H-012 | 2026-08-10 | Coint | Vol-aware entry \(k_t\) vs S1 vol targeting         | —             | NOT IMPLEMENTED |


---



## H-001 · Coint · Universes A / B / C · 2026-08-10


| Field                  |                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                    |
| **What it is**         | Compare candidate pair universes A, B, and C under the same trading rules.                                                                         |
| **Hypothesis**         | Evaluate universes A, B, and C on Sharpe (after costs), max DD, and correlation to S1.                                                             |
| **Economic rationale** | —                                                                                                                                                  |
| **Data required**      | —                                                                                                                                                  |
| **Test to complete**   | Validation protocol: WF fold-val boxplots of Sharpe, max DD, corr to S1; freeze universe on research IS; one sealed OOS tearsheet.                 |
| **Notes**              | Lock chosen universe before later hyps; do not re-pick after downstream bake-offs.                                                                 |


---



## H-002 · Coint · 1D vs 4H / 1H after costs · 2026-08-10


| Field                  |                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                    |
| **What it is**         | Same pair logic on daily vs 4H vs 1H bars, with timeframe-appropriate frictions.                                                                   |
| **Hypothesis**         | 1D timeframe survives costs better than 4H/1H after frictions.                                                                                     |
| **Economic rationale** | —                                                                                                                                                  |
| **Data required**      | —                                                                                                                                                  |
| **Test to complete**   | Compare on Sharpe, max DD, corr to S1 under the Validation protocol (identical pair set and entry rule where possible).                            |
| **Notes**              | Compare on Sharpe, max DD, corr to S1 under the Validation protocol.                                                                               |


---



## H-003 · Coint · Kalman β vs static OLS hedge · 2026-08-10


| Field                  |                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                 |
| **What it is**         | Time-varying hedge ratio via Kalman filter vs static OLS β; spread built point-in-time per top Notes.                                                                                           |
| **Hypothesis**         | Kalman β beats static OLS hedge on OOS spread Sharpe / max DD / corr to S1.                                                                                                                     |
| **Economic rationale** | —                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                               |
| **Test to complete**   | Traditional z-score entry/exit for both arms. Score candidates on fold-val (not train Sharpe). WF boxplots of Sharpe, max DD, corr to S1; sealed OOS once.                                      |
| **Notes**              | Entry/exit = traditional z-score. Spread history obeys PIT β Notes. Isolates hedge quality before gates or alternate scores. **Uses the single shared Kalman module** (create here if not yet present). Score candidates on fold-val (not train Sharpe). |


---



## H-004 · Coint · Half-life gate · 2026-08-10


| Field                  |                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                 |
| **What it is**         | Only trade pairs whose estimated mean-reversion half-life lies in a pre-registered band \([L_{\min}, L_{\max}]\).                                                                               |
| **Hypothesis**         | Half-life gate (only trade if half-life ∈ [Lmin, Lmax]) improves Sharpe / max DD / corr to S1.                                                                                                  |
| **Economic rationale** | —                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                               |
| **Test to complete**   | Same traditional z-score entry as H-003. Primary band vs at most one alternate via WF val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                               |
| **Notes**              | Same traditional z-score entry as H-003; adds half-life clipping only. **Pre-register one** \([L_{\min}, L_{\max}]\) from literature or economics as the primary band; allow **at most one** alternate band for robustness — **not** a grid over many bands. Choose between primary vs alternate (if any) via WF val boxplots; sealed OOS once. |


---



## H-005 · Coint · Entry extremity scores · 2026-08-10


| Field                  |                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                 |
| **What it is**         | Bake-off of separable alternatives to fixed traditional z for deciding when the spread is overbought/oversold. Implement only chosen subsets.                                                   |
| **Hypothesis**         | Entry scores other than fixed traditional z improve OOS Sharpe / max DD / corr to S1 after costs (variants tested separately).                                                                  |
| **Economic rationale** | —                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                               |
| **Test to complete**   | Bake-off chosen subsets under the Validation protocol; same pairs/costs/exits where possible. Metrics: Sharpe, max DD, corr to S1.                                                              |
| **Notes**              | Selectable variants (not one forced stack): (1) Rolling / EWM z + asymmetric bands — enter at \(\pm k_{\text{in}}\), exit nearer mean or at \(k_{\text{out}} < k_{\text{in}}\); rolling vs EWM vol. (2) OU / AR(1) residual score — see below. (3) Fused Kalman β + HMM regime + Kalman-on-spread innovation — β from shared Kalman; HMM gates mean-reverting vs trending/broken (flat when not MR); in MR only, trade standardized Kalman innovation on the spread (**reuse shared Kalman module** for β and spread state; no second KF core). (4) Copula / conditional quantile — see below. |


**OU / AR(1) residual score (A-level Maths)**

- A **pair spread** is one series after hedging (e.g. \(s_t = y_t - \beta x_t\)).
- Fit **AR(1)** on that spread: today’s spread ≈ pull toward a mean + leftover. **OU** = continuous-time twin.
- **Residual** = leftover / distance of \(s_t\) from \(\mu\) — from **spread on its own past**, not stock-on-stock (β already done).
- **Score:** leftover size vs model noise → overbought/oversold under mean reversion.

Steps: (1) build PIT spread (2) estimate \(\mu, \phi, \sigma\) on past only (3) distance of \(s_t\) from \(\mu\) in units of \(\sigma\) (4) enter beyond threshold.

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



## H-006 · Coint · ADX and/or RSI trend filter · 2026-08-10


| Field                  |                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                    |
| **What it is**         | Require ADX and/or RSI confirmation (or veto) before opening a spread trade.                                                                       |
| **Hypothesis**         | ADX and/or RSI confirmation/veto before entry improves net Sharpe / max DD / corr to S1 vs ungated entries.                                        |
| **Economic rationale** | —                                                                                                                                                  |
| **Data required**      | —                                                                                                                                                  |
| **Test to complete**   | Validation protocol: fold-val boxplots of Sharpe, max DD, corr to S1; sealed OOS once.                                                             |
| **Notes**              | Prefer defaults (e.g. RSI 14, ADX 14) + at most a tiny pre-registered robustness set. Select via Validation protocol (fold-val boxplots; sealed OOS once). |


---



## H-007 · Coint · Exit: n half-lives + ATR SL + max-loss breaker · 2026-08-10


| Field                  |                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                 |
| **What it is**         | Time-stop after \(n\) half-lives; ATR-on-spread stop sized to fixed 1% portfolio loss; per-pair max-loss circuit breaker.                                                                        |
| **Hypothesis**         | Time-stop after \(n\) half-lives, ATR-on-spread stop sized to fixed 1% loss, plus a per-pair max-loss circuit breaker, improves Sharpe / max DD / corr to S1 vs mean-exit-only.                |
| **Economic rationale** | —                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                               |
| **Test to complete**   | Compare to mean-exit-only under the Validation protocol (Sharpe, max DD, corr to S1). Discrete knobs (e.g. ATR multiple) scored on fold-val, not fold-train Sharpe.                              |
| **Notes**              | **Time exit:** flat if not reverted within \(n \times\) half-life. **ATR stop:** position size so stop hit ≈ **1%** portfolio loss. **Max-loss circuit breaker:** if a **single pair’s** open PnL reaches a hard floor (example **−20%** — exact equity attribution fixed at implement), **liquidate that pair immediately** and do not re-enter until a reset rule. Catastrophe backstop separate from the 1% ATR risk unit. |


---



## H-008 · Coint · Cointegration-break flat rule · 2026-08-10


| Field                  |                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                    |
| **What it is**         | Go flat when a cointegration / stationarity break is detected on the live spread.                                                                  |
| **Hypothesis**         | Cointegration-break flat rule improves max DD (and Sharpe / corr to S1) more than it hurts return.                                                 |
| **Economic rationale** | —                                                                                                                                                  |
| **Data required**      | —                                                                                                                                                  |
| **Test to complete**   | Validation protocol: Sharpe, max DD, corr to S1 on fold-val boxplots; sealed OOS once.                                                             |
| **Notes**              | —                                                                                                                                                  |


---



## H-009 · Coint · Overlapping legs: allow vs never-allow · 2026-08-10


| Field                  |                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                 |
| **What it is**         | Bake-off **allow overlapping legs** vs **never allow overlapping legs** (shared ticker across pairs).                                                                                           |
| **Hypothesis**         | Forbidding overlapping legs improves portfolio Sharpe / max DD / corr to S1 vs allowing overlaps, after costs.                                                                                  |
| **Economic rationale** | —                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                               |
| **Test to complete**   | Allow vs never-allow under the Validation protocol (Sharpe, max DD, corr to S1).                                                                                                                |
| **Notes**              | **Never-allow rules:** Conflict = candidate shares **any ticker** with an already open pair, or with another candidate on the same bar. If a position is **already open** and a new signal shares a leg → **do not open** the new position. If **two** (or more) signals would open on the **same bar** and their legs overlap → open only the candidate with the best **Score × confidence** (\(|\text{score}| \times (1 - p_{\text{ADF}})\)); leave the other(s) flat. Does **not** use spread correlation (that is H-010). Report concentration / shared-leg exposure under the allow arm. |


---



## H-010 · Coint · Kalman spread-correlation gate · 2026-08-10


| Field                  |                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                 |
| **What it is**         | Block new pairs when Kalman-filtered correlation of spreads exceeds a threshold \(k\); same conflict resolution as H-009.                                                                       |
| **Hypothesis**         | Blocking new pairs when \(|\hat\rho_t| > k\) improves Sharpe / max DD / corr to S1 vs no correlation gate; test whether cancel / do not open correlated conflicts is worth keeping.             |
| **Economic rationale** | —                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                               |
| **Test to complete**   | Bake-off no corr gate vs gate on; pre-register a small \(k\) set; choose on fold-val boxplots (Sharpe, max DD, corr to S1); sealed OOS once.                                                    |
| **Notes**              | Estimate **time-varying correlation** between pair spread series (prefer spread **returns**/changes) with a **Kalman filter** tracking \(\hat\rho_t\), **not** a fixed-window OLS/sample correlation. **Must use the single shared Kalman module** (state = correlation or bivariate moments → \(\hat\rho_t\)); no new KF core. If \(|\hat\rho_t| > k\) between a new candidate and an open (or same-bar) pair, treat as a **conflict** and apply the **same resolution rules as H-009**. Independent of shared-leg rules; this hyp isolates the corr gate. |


---



## H-011 · Coint · Score × confidence sizing · 2026-08-10


| Field                  |                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                 |
| **What it is**         | Position size proportional to \(\|\text{score}\| \times (1 - p_{\text{ADF}})\). **Score** = entry extremity from the chosen H-005 variant (or traditional z when that is the active entry).      |
| **Hypothesis**         | Sizing \(\propto \|\text{score}\| \times (1 - p_{\text{ADF}})\) beats equal risk or score-only sizing on Sharpe / max DD / corr to S1.                                                           |
| **Economic rationale** | —                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                               |
| **Test to complete**   | Arms: equal risk vs score-only vs score×(1−p). Validation protocol boxplots; sealed OOS once.                                                                                                   |
| **Notes**              | ADF \(p\) = stationarity/coint evidence; \((1-p)\) up-weights when stronger. Same Score × confidence formula is the **tie-break / priority metric** in H-009 and H-010 conflict rules; this hyp tests it as a **position-size** multiplier. Selection via Validation protocol. |


---



## H-012 · Coint · Vol-aware entry \(k_t\) vs S1 vol targeting · 2026-08-10


| Field                  |                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | NOT IMPLEMENTED                                                                                                                                                                                 |
| **What it is**         | Vol-aware entry threshold \(k_t = k_0 \cdot \sigma_t / \bar\sigma\) (need a larger move in high-vol regimes) vs fixed \(k\), compared to S1-style portfolio vol targeting.                        |
| **Hypothesis**         | \(k_t = k_0 \cdot \sigma_t / \bar\sigma\) beats fixed \(k\) on Sharpe / max DD / corr to S1; compare in-test to S1 portfolio vol targeting (S1 H-014 / `06_risk/vol_targeting.py`).              |
| **Economic rationale** | —                                                                                                                                                                                               |
| **Data required**      | —                                                                                                                                                                                               |
| **Test to complete**   | Arms: fixed \(k\) vs vol-aware entry \(k_t\) vs S1-style portfolio VT. Validation protocol; sealed OOS once.                                                                                    |
| **Notes**              | Selection via Validation protocol. Entry-threshold lever vs position-leverage lever.                                                                                                            |


**Formulae**

- Vol-aware entry: \(k_t = k_0 \cdot \sigma_t / \bar\sigma\)
- S1-style vol targeting: \(L_t \propto \sigma_{\text{target}} / \hat\sigma_{\text{portfolio}}\)
