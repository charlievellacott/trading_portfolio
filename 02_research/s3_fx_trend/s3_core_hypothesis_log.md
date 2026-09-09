# S3 FX Trend — Core Sleeve Hypothesis Log

## Validation protocol

Mirror S2 Option 4 (institutional hybrid): purged expanding walk-forward on **research IS** → per-arm fold-val tables and boxplots → **discretionary STAR** → full-IS report for the frozen arm → **one** sealed OOS tearsheet.

**Core rules:** fold-val scores **pre-registered** arms only; STAR is human discretionary; sealed OOS once after STAR; maintain a **variant ledger** for the core sleeve. Never tune on sealed OOS.

**Inference columns (secondary, not for STAR):** `psr` = P(true SR > 0); `dsr_local`; `dsr_stack`.

**Primary metrics (every hyp):** net **Sharpe**, **max drawdown**, **corr to S1**, and **corr to S2** on the same return series. Rank on net Sharpe; report correlations beside it — **no combined score**. Negative corr to S1/S2 beats low positive.

**Fair branch rule:** freeze the trunk (`HORIZON_STAR`, `RISK_STAR`, `FAMILY_STAR`, …) before overlay bake-offs. H-010 / H-011 / plain core use the **same** frozen stack and **equal coarse arm budgets**. Deep tune **only** the winning overlay in H-012.

**Notebooks:** `02_research/s3_fx_trend/notebooks/core_sleeve/`.

---

### Timing contract (all core hyps)

Same spirit as S2 — **not** same-bar close-fill; **not** S1 trade-date / `feature_date` indexing.

| Role | Timestamp |
| ---- | --------- |
| Features / signal / decision | Close of bar `t` (pre-registered FX day boundary) |
| Optional macro / alt | After close `t`, before open `t+1` (if knowable then) |
| Orders | Queued just before open `t+1` |
| Fill | Open of `t+1` (next tradable open after the defined close) |
| First PnL | From open `t+1` onward |

**Day boundary:** pre-register in algorithm notes (recommend CTA-style **17:00 America/New_York**, or freeze broker/OANDA candle TZ). Every daily TSMOM lookback, Donchian high/low, and `t+1` open uses that clock; changing it later invalidates STARs.

Do **not** add an extra `.shift(1)` on close-`t` features. Do **not** store next-bar open on the signal row as a feature.

**Returns:** simple (percentage) returns for signals, vol, and PnL unless a hyp explicitly states otherwise.

---

### Gate vs tilt (H-010 / H-011)

| | **Gate** | **Tilt** |
| --- | --- | --- |
| Meaning | Binary permission | Continuous size modifier |
| Macro agrees with trend | Allow trade / keep risk | Scale risk **up** |
| Macro disagrees | **Flat / block new entry** | Scale risk **down** (may still hold small risk) |

Carry and value are tested as **overlays on frozen trend**, not as replacement primary signals.

---

### Stack order (do not reorder)

H-001 existence → H-002 horizon → H-003–H-005 risk → H-006 family → H-007 breakout definition (if needed) → H-008–H-009 hygiene → H-010–H-011 overlays → H-012 winner polish → H-013 turnover control.

---


| ID    | Date       | Asset | Factor                                              | Data required                                      | Status            |
| ----- | ---------- | ----- | --------------------------------------------------- | -------------------------------------------------- | ----------------- |
| H-001 | 2026-09-08 | FX    | Single-horizon daily TSMOM clears costs             | G10 daily OHLCV + OANDA cost model                 | NOT IMPLEMENTED   |
| H-002 | 2026-09-08 | FX    | Horizon blend 1M/3M/12M vs singles                  | Same                                               | NOT IMPLEMENTED   |
| H-003 | 2026-09-08 | FX    | Pair vol targeting vs raw ±1                        | Same + vol-targeting modules                       | NOT IMPLEMENTED   |
| H-004 | 2026-09-08 | FX    | Portfolio vol targeting vs pair-only                | Same                                               | NOT IMPLEMENTED   |
| H-005 | 2026-09-08 | FX    | Delever/flat on pair and/or book vol spike          | Same                                               | NOT IMPLEMENTED   |
| H-006 | 2026-09-08 | FX    | Trend-family bake-off (TSMOM / Donchian / both)     | Same                                               | NOT IMPLEMENTED   |
| H-007 | 2026-09-08 | FX    | Donchian H/L definition / session filters           | Same (+ session timestamps if required)            | NOT IMPLEMENTED   |
| H-008 | 2026-09-08 | FX    | Conviction scaling (\|signal\|) vs sign-only          | Same                                               | NOT IMPLEMENTED   |
| H-009 | 2026-09-08 | FX    | Currency / corr conflict + per-currency exposure caps | Same                                               | NOT IMPLEMENTED   |
| H-010 | 2026-09-08 | FX    | Carry gate or tilt (weekly rescore)                 | Rate differentials + daily book                    | NOT IMPLEMENTED   |
| H-011 | 2026-09-08 | FX    | Value (PPP / real rate) gate or tilt                | PPP / real-rate series + daily book                | NOT IMPLEMENTED   |
| H-012 | 2026-09-08 | FX    | Winner polish (chosen overlay only)                 | Frozen overlay stack                               | NOT IMPLEMENTED   |
| H-013 | 2026-09-08 | FX    | Banded rebalance / weak-signal flat                 | Frozen core (+ overlay if any)                     | NOT IMPLEMENTED   |


---

## H-001 · FX · Single-horizon daily TSMOM clears costs · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Literature-faithful **12M** daily time-series momentum on each G10 pair with **raw ±1** weights (no inverse-vol), under realistic **OANDA** costs. |
| **Hypothesis**         | 12M TSMOM earns **after-cost** median fold-val net Sharpe **> 0** on the G10 book. |
| **Economic rationale** | Slow incorporation of macro news, stop/CTA feedback, and persistent FX trends create serial correlation; the premium should survive G10 friction if the sleeve is viable. |
| **Data required**      | Daily G10 OHLCV on the pre-registered day boundary; OANDA cost model. |
| **Test to complete**   | Validation protocol on research IS; primary arm = 12M TSMOM, sign → ±1. Report 1M and 3M as **diagnostics only** (same costs/timing) — **do not** write them into the star stack or use them to pick `HORIZON_STAR`. |
| **Notes**              | Existence / kill gate for the sleeve. Horizon bake-off is **H-002** (all singles + blend as pre-registered arms). Notebook: `notebooks/core_sleeve/H-001_tsmom_baseline.ipynb` (planned). |


**Kill / continue**

| Band | Median fold-val net Sharpe | Action |
| ---- | -------------------------- | ------ |
| Fail | **≤ 0** | Shelve or redesign before H-002+ |
| Pass | **> 0** | Continue to H-002 |

**Signal (primary):** \(\mathrm{sign}(R_{t-252:t})\) or equivalent 12M simple return ending at close \(t\) (document skip month if used). Position = ±1 per pair until later hyps.

---

## H-002 · FX · Horizon blend vs singles · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Bake-off of single-horizon TSMOM vs equal blend of 1M / 3M / 12M under the same raw ±1 (or soft default) sizing and costs. |
| **Hypothesis**         | `equal_blend` improves **Sharpe and max DD** vs the best single among the pre-registered singles. |
| **Economic rationale** | Multi-horizon trend is standard in CTA literature; blending reduces dependence on one speed and can cut chop at a single lookback. |
| **Data required**      | Same as H-001. |
| **Test to complete**   | Arms: `1M` \| `3M` \| `12M` \| `equal_blend`. Fold-val boxplots; freeze `HORIZON_STAR`. Do **not** define “best single” from H-001 diagnostics — all singles live in this screen. |
| **Notes**              | Notebook: `notebooks/core_sleeve/H-002_horizon_blend.ipynb` (planned). |


---

## H-003 · FX · Pair vol targeting · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Inverse-vol (or target-vol) sizing **per pair** vs raw ±1, under frozen `HORIZON_STAR`. |
| **Hypothesis**         | Pair vol targeting improves Sharpe **and** max DD vs raw ±1. |
| **Economic rationale** | G10 pairs have unequal volatility; ±1 overweights noisy pairs and understates risk contribution of quiet ones. |
| **Data required**      | Same + existing vol-targeting modules (`risk.analytics` / S1-style helpers — reuse, do not fork). |
| **Test to complete**   | Arms: `off` \| `pair_vt`. Freeze contribution to `RISK_STAR`. |
| **Notes**              | Use current project VT modules. Notebook: `notebooks/core_sleeve/H-003_pair_vol_target.ipynb` (planned). |


---

## H-004 · FX · Portfolio vol targeting · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Scale the **book** to a portfolio vol target on top of pair sizing. |
| **Hypothesis**         | Portfolio VT improves book Sharpe and/or max DD vs pair-only sizing. |
| **Economic rationale** | Pair VT equalises contribution; book VT stabilises aggregate risk as correlations and breadth change. |
| **Data required**      | Same as H-003. |
| **Test to complete**   | Arms: `pair_only` \| `pair_plus_book_vt` under frozen horizon + pair VT choice. |
| **Notes**              | Notebook: `notebooks/core_sleeve/H-004_book_vol_target.ipynb` (planned). |


---

## H-005 · FX · Vol-spike delever / flat · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | When pair and/or book realised vol spikes vs a trailing baseline, delever or go flat. |
| **Hypothesis**         | Spike policy improves max DD without destroying net Sharpe vs `RISK_STAR` without the policy. |
| **Economic rationale** | Trend books bleed or gap in vol regimes; cutting risk into spikes improves Calmar-like outcomes. |
| **Data required**      | Same. |
| **Test to complete**   | Arms: `off` \| `pair_delever` \| `book_delever` \| `both`. Complete `RISK_STAR`. |
| **Notes**              | Notebook: `notebooks/core_sleeve/H-005_vol_spike.ipynb` (planned). |


---

## H-006 · FX · Trend-family bake-off · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Compare trend sensors under **frozen** `HORIZON_STAR` + `RISK_STAR`. |
| **Hypothesis**         | One of TSMOM, classic Donchian, or both wins on holdout-style fold-val with a robustness plateau. |
| **Economic rationale** | Donchian breakouts capture level-driven continuation and natural flat zones; TSMOM is always-in; combining uses breakout as permission on TSMOM direction. |
| **Data required**      | Same. |
| **Test to complete**   | Arms: (1) **TSMOM** (2) **classic Donchian only** (3) **both** (TSMOM direction × Donchian permission). Baseline Donchian highs/lows = pre-registered **day-boundary** bars (e.g. NY-close). Small N grid **or** freeze one N for the bake-off (prefer freeze one N if grid would confound family choice). Freeze `FAMILY_STAR`. |
| **Notes**              | No ATR-channel arm in this screen. Session / H-L definition variants are **H-007**. Notebook: `notebooks/core_sleeve/H-006_trend_family.ipynb` (planned). |


---

## H-007 · FX · Donchian H/L definition / session filters · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | What counts as the breakout high/low once a Donchian-using family is chosen. |
| **Hypothesis**         | Session-aware H/L definition improves Sharpe and/or DD vs naive day-boundary Donchian (or is flat — then keep baseline). |
| **Economic rationale** | Asia-only spikes can print false extremes on EUR/GBP; continuation is likelier when breaks occur in home liquidity (London–NY). |
| **Data required**      | Same; session timestamps if arms (b)/(c) need them. |
| **Test to complete**   | **Skip / N/A** if `FAMILY_STAR` is TSMOM-only. Else arms: (a) NY-close daily H/L (baseline); (b) require London–NY overlap confirmation; (c) ignore Asia-only extremes on EUR/GBP (and similar). Freeze `BREAKOUT_DEF_STAR`. |
| **Notes**              | Runs **after** H-006 so family choice is not confounded with session definition. Notebook: `notebooks/core_sleeve/H-007_donchian_sessions.ipynb` (planned). |


---

## H-008 · FX · Conviction scaling · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Weight by \(\lvert signal \rvert\) (strength) vs sign-only, under frozen family (+ breakout def if any). |
| **Hypothesis**         | Conviction scaling improves Sharpe and/or DD vs sign-only. |
| **Economic rationale** | Weak trends are noise; stronger measured trend warrants more risk. |
| **Data required**      | Same. |
| **Test to complete**   | Arms: `sign_only` \| `abs_scale` (define normalization so leverage stays comparable under `RISK_STAR`). |
| **Notes**              | Notebook: `notebooks/core_sleeve/H-008_conviction.ipynb` (planned). |


---

## H-009 · FX · Currency / corr conflict + exposure caps · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Skip new risk if highly correlated with the open book **or** if adding the pair would breach a **per-currency** gross/net exposure cap. |
| **Hypothesis**         | Conflict / currency caps improve diversification (max DD, kurtosis) without destroying net Sharpe. |
| **Economic rationale** | Equal long/short **pair count** does not control USD (or other) factor load; G10 pairs share currencies. Caps limit concentrated currency risk. Hard USD-neutral projection is **out of scope** for v1 (optional later). |
| **Data required**      | Same. |
| **Test to complete**   | Arms: `off` \| `corr_gate` \| `currency_cap` \| `both`. Pre-register corr threshold and cap levels (small set). |
| **Notes**              | Notebook: `notebooks/core_sleeve/H-009_currency_corr.ipynb` (planned). |


---

## H-010 · FX · Carry gate or tilt · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Interest-rate **differential** as a **gate** or **tilt** on the frozen trend book; **weekly** rescore on daily bars (no weekly price resample required for TSMOM). |
| **Hypothesis**         | Carry gate or tilt improves Sharpe and/or DD vs core alone under equal arm budget. |
| **Economic rationale** | Carry is a slow FX premium; using it as permission/tilt avoids fitting carry as a standalone always-in signal. |
| **Data required**      | Rate differentials (vendor TBD — document before coding); daily G10 book. |
| **Test to complete**   | Arms: `off` \| `gate` \| `tilt`. Optional breakout confirm **only if** `FAMILY_STAR` already uses Donchian. Equal coarse budget with H-011. Candidate for `OVERLAY_STAR`. |
| **Notes**              | **Data path TBD** before implementation. Not a new primary strategy. Notebook: `notebooks/core_sleeve/H-010_carry_overlay.ipynb` (planned). |


---

## H-011 · FX · Value (PPP / real rate) gate or tilt · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | PPP / real-rate **value** as gate or tilt; **monthly** rescore (quarterly as robustness), daily execution book. |
| **Hypothesis**         | Value gate or tilt improves Sharpe and/or DD vs core alone under the same budget as H-010. |
| **Economic rationale** | Real mispricing mean-reverts slowly; as overlay it leans the trend book without daily value noise. |
| **Data required**      | PPP / real-rate series (vendor TBD); daily G10 book. |
| **Test to complete**   | Arms: `off` \| `gate` \| `tilt`. Same fairness rules as H-010. Candidate for `OVERLAY_STAR`. |
| **Notes**              | **Data path TBD** before implementation. Notebook: `notebooks/core_sleeve/H-011_value_overlay.ipynb` (planned). |


**Overlay selection:** compare H-010 / H-011 arms and plain core on the frozen trunk; pick `OVERLAY_STAR` ∈ {`none`, carry_gate, carry_tilt, value_gate, value_tilt, …}.

---

## H-012 · FX · Winner polish (overlay only) · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | **One small grid** on the **chosen overlay** only (research IS). |
| **Hypothesis**         | Modest threshold / scale polish improves fold-val metrics without reopening the trunk. |
| **Economic rationale** | Overlay strength knobs are second-order; equal budgets at H-010/H-011 already selected structure. |
| **Data required**      | Frozen `OVERLAY_STAR` stack. |
| **Test to complete**   | If `OVERLAY_STAR = none`, skip or document no polish. If carry/value **gate**: 1–2 agree-rule / strength thresholds. If **tilt**: 2–3 agree/disagree scale pairs (e.g. 1.25/0.75 vs 1.5/0.5). |
| **Notes**             | **Forbidden here:** re-grid TSMOM horizons, Donchian N, session arms, VT targets, currency caps, banded-rebalance bands, core hold durations. Those live in H-002 / H-006–H-007 / H-003–H-005 / H-009 / H-013. Notebook: `notebooks/core_sleeve/H-012_overlay_polish.ipynb` (planned). |


---

## H-013 · FX · Banded rebalance / weak-signal flat · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Implementation control on turnover after the signal + risk + overlay stack is frozen. |
| **Hypothesis**         | Banded rebalance and/or weak-signal flat cut turnover/cost drag without destroying holdout / fold-val Sharpe. |
| **Economic rationale** | Daily sign flips and tiny weight wiggles pay spread without economic change in stance. |
| **Data required**      | Frozen core (+ overlay if any). |
| **Test to complete**   | Arms include: **banded rebalance** — trade only when \(\lvert w^{\mathrm{target}} - w^{\mathrm{current}} \rvert\) exceeds a band (pre-register e.g. 10% / 20% of target scale); **weak-signal flat** — if \(\lvert signal \rvert\) below threshold → weight 0. Compare turnover, net Sharpe, max DD. |
| **Notes**              | Notebook: `notebooks/core_sleeve/H-013_rebalance_bands.ipynb` (planned). |


---

## Desk checklist (core)

- [ ] H-001 through H-013 logged above (this file)
- [ ] Timing + day boundary written into algorithm notes / overview
- [ ] Variant ledger path for core sleeve registered when backtest harness exists
- [ ] Carry / value data vendors chosen before H-010 / H-011 coding
- [ ] Event sleeve remains in [`s3_event_hypothesis_log.md`](s3_event_hypothesis_log.md) — do not mix into core STAR
