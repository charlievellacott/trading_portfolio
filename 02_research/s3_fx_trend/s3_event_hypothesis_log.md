# S3 FX Trend — Event Sleeve Hypothesis Log

Deferred relative to the **core** sleeve. Do not run E-* STAR decisions until the core trunk (and preferred overlay, if any) is frozen in [`s3_core_hypothesis_log.md`](s3_core_hypothesis_log.md), unless explicitly doing isolated data/vendor work that cannot leak into core selection.

## Validation protocol

Same institutional hybrid as core / S2: purged expanding walk-forward on **research IS** → fold-val arms → discretionary STAR → full-IS → **one** sealed OOS. Separate **variant ledger** for the event sleeve.

**Inference (secondary):** `psr` = P(true SR > 0); `dsr_local`; `dsr_stack`.

**Primary metrics:** net Sharpe, max DD, corr to S1, corr to S2 (and corr to **frozen core** book when comparing overlays).

**Notebooks:** `02_research/s3_fx_trend/notebooks/event_sleeve/`.

---

### Timing / clocks

- **Event trigger** = economic release print time τ (not London open).
- **E-003 bake-off:** `daily_post_event` \| `1h` only — **floor is 1h** (no 5m / 1m). Winner → `EVENT_BAR_STAR`.
- **Fill rule (both arms):** open of the **first bar whose open is strictly after τ** on the chosen grid (`map_event_to_next_bar_fill`). Surprise is known at τ; never enter on a bar that opened at or before τ.
- **Core contract** (close \(t\) → open \(t+1\) on NY 17:00 days) still applies when the event bar is daily.
- Do **not** default to hold-until-next-news.

**Data caveat (CSV):** research calendar
`01_data/data_files/s3_fx_trend/economic_calendar.csv` is **not vintage-safe** —
`actual` / `forecast` / `previous` can reflect later revisions. Use for research /
diagnostics only; do **not** paper-trade events off this CSV alone (see
`s3_algorithm_notes.md` · Live Event Calendar Viability).

**Quote convention:** map surprise onto each pair so “stronger currency X” has the correct sign on BASE/QUOTE. If X is base → positive surprise → long bias; if X is quote → flip sign. Without this, EURXXX and XXXEUR disagree on the same macro surprise.

---

### Standardization (pre-registered — not Sharpe-screened)

Raw surprise \(s = \text{actual} - \text{forecast}\).

Standardize **per release series** (e.g. US NFP over time), not a separate \(\sigma\) per pair:

\[
z = \frac{s}{\hat\sigma_{N,\text{release}}}
\]

- **\(N = 20\)** trailing surprises of that **same** release (pre-registered; do not grid \(N\) on IS Sharpe).
- Min history: if fewer than \(N\) prior prints, use expanding history until \(N\) is available (or skip until warm).
- Then map \(z\) onto affected pairs via **quote convention**.

**Gate \(k\):** one **book-level** threshold shared across pairs. Arms \(\{0.5, 1.0, 1.5\}\); freeze `K_STAR`. No per-pair or per-currency \(k\) in v1.

---


| ID    | Date       | Asset | Factor                                                         | Data required                                | Status          |
| ----- | ---------- | ----- | -------------------------------------------------------------- | -------------------------------------------- | --------------- |
| E-001 | 2026-09-08 | FX    | Signed standardized surprise + \|z\| gate                      | Forecast calendar + G10 prices + OANDA costs | NOT IMPLEMENTED |
| E-002 | 2026-09-08 | FX    | Impact tier (high / high+mid / all)                            | Event impact tags + E-001 stack              | NOT IMPLEMENTED |
| E-003 | 2026-09-08 | FX    | Bar size: daily post-event vs intraday                         | Vendor 1H (+ yfinance validation window)     | NOT IMPLEMENTED |
| E-004 | 2026-09-08 | FX    | Hold / exit rules                                              | Frozen E bar + signal                        | NOT IMPLEMENTED |
| E-005 | 2026-09-08 | FX    | Standalone event vs confirm-only with core family              | Frozen core `FAMILY_STAR` + event stack      | NOT IMPLEMENTED |
| E-006 | 2026-09-08 | FX    | Size ∝ \|z\| (scale 0.5–2)                                       | Frozen E-001…E-005 choices                   | NOT IMPLEMENTED |


---

## E-001 · FX · Signed surprise + \|z\| gate · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Economic surprise trading: signed \(z\) via quote convention, optional Alphalens/IC diagnosis, then after-cost backtest with a shared \|z\| gate. |
| **Hypothesis**         | Standardized signed surprise has after-cost edge on affected G10 pairs; a global \|z\| > k gate improves net Sharpe / DD vs ungated always-trade-on-print. |
| **Economic rationale** | Markets reprice macro news with underreaction / overreaction; surprises are comparable only after per-release standardization. |
| **Data required**      | Consensus forecast + actual prints; G10 OHLCV; OANDA costs. |
| **Test to complete**   | **Part A (diagnostic):** Alphalens / IC-style analysis of signed surprise vs forward returns at candidate horizons — **supporting only**, not STAR. **Part B (decision):** backtest with \(N=20\) per-release \(\sigma\); quote-convention mapping; arms `k ∈ {0.5, 1.0, 1.5}` (+ optional ungated diagnostic). Freeze `K_STAR` (and surprise pipeline). |
| **Notes**              | Do not pick per-pair \(k\). Do not Sharpe-grid \(N\). Notebook: `notebooks/event_sleeve/E-001_surprise_gate.ipynb` (planned). |


---

## E-002 · FX · Impact tier · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Restrict the calendar to high-impact only vs high+medium vs all tagged events. |
| **Hypothesis**         | High-only (or high+mid) beats trading all events after costs. |
| **Economic rationale** | Low-impact prints add turnover and noise; high-impact surprises drive FX. |
| **Data required**      | Vendor impact classification + frozen E-001 pipeline. |
| **Test to complete**   | Arms: `high` \| `high_mid` \| `all` under frozen `K_STAR`. |
| **Notes**              | Notebook: `notebooks/event_sleeve/E-002_impact_tier.ipynb` (planned). |


---

## E-003 · FX · Bar size (daily vs intraday) · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Decision/execution on **daily** post-event bars vs **1h** around the print, after costs. Floor: no sub-1h. |
| **Hypothesis**         | One bar size wins on net Sharpe / DD; OANDA 1H is faithful enough vs yfinance where overlap exists. |
| **Economic rationale** | Surprise alpha may be fast (hours) or persist into the next daily session; costs rise with finer bars. |
| **Data required**      | OANDA 1D + 1H; yfinance 1H for overlap validation. |
| **Test to complete**   | (1) Vendor validation notebook. (2) Arms: `daily_post_event` \| `1h` under frozen E-001/E-002. Fill = next-bar open after τ on that grid. Freeze `EVENT_BAR_STAR`. |
| **Notes**              | Notebook: `notebooks/event_sleeve/E-003_bar_size.ipynb` (planned). |


---

## E-004 · FX · Hold / exit · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **What it is**         | How long to keep event risk on after entry. |
| **Status**             | NOT IMPLEMENTED |
| **Hypothesis**         | A pre-registered exit rule beats hold-until-next-news on net Sharpe / DD. |
| **Economic rationale** | Event drift is finite; next unrelated print is not an economic exit. |
| **Data required**      | Frozen bar size + signal. |
| **Test to complete**   | Arms: fixed hold `{1d, 5d, 10d}` (adjust units if E-003 selects intraday, e.g. fixed hours from IC diagnosis); exit when short trend/breakout flips; ATR or time-stop. **Not** hold-until-next-news. |
| **Notes**              | If Part A IC in E-001 strongly favours intraday horizons, shift fixed-hold grid to matching units **pre-register before the screen** — do not freely expand after seeing Sharpe. Notebook: `notebooks/event_sleeve/E-004_hold_exit.ipynb` (planned). |


---

## E-005 · FX · Standalone vs confirm-only · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Event book as standalone overlay vs risk only when surprise **agrees** with frozen core daily family (`FAMILY_STAR`). |
| **Hypothesis**         | Confirm-only improves DD / corr properties vs standalone, or standalone wins on Sharpe — pick explicitly. |
| **Economic rationale** | Events that fight the prevailing trend may be noise; confirmation reduces two-way churn with the core sleeve. |
| **Data required**      | Frozen core family signal + frozen event stack. |
| **Test to complete**   | Arms: `standalone` \| `confirm_only`. Report corr to core. |
| **Notes**              | Notebook: `notebooks/event_sleeve/E-005_core_confirm.ipynb` (planned). |


---

## E-006 · FX · Size ∝ \|z\| · 2026-09-08


| Field                  | |
| ---------------------- | --- |
| **Status**             | NOT IMPLEMENTED |
| **What it is**         | Position scale based on surprise magnitude after direction / gate / hold / integration are frozen. |
| **Hypothesis**         | Scaling risk with \|z\| (clipped) beats flat size when \|z\| > k. |
| **Economic rationale** | Larger standardized surprises warrant more risk; weak prints near the gate are marginal. |
| **Data required**      | Frozen E-001…E-005. |
| **Test to complete**   | Arms: `flat` (unit size if pass gate) \| `scale_0.5_to_2` — map \|z\| (above k) linearly into leverage mult \(\in [0.5, 2]\) with a pre-registered cap rule; keep book risk comparable via existing VT if used. |
| **Notes**              | Prefer magnitude→mult with book risk control over full-G10 CS rank of \|z\| (many pairs have no event). Optional later: rank only within the **active event cohort** that timestamp. Notebook: `notebooks/event_sleeve/E-006_surprise_sizing.ipynb` (planned). |


---

## Desk checklist (event)

- [ ] E-001 through E-006 logged above (this file)
- [ ] Forecast / calendar vendor chosen; quote-convention table for G10 documented
- [ ] \(N=20\) and shared \(k\) grid respected (no per-pair \(k\), no Sharpe-grid on \(N\))
- [ ] Core STAR frozen before E-005 confirm-only bake-off
- [ ] Separate variant ledger from core
