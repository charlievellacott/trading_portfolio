# S2 v2 hypothesis log — PIT S&P pairs

Version 1 (letter universes, including E) stays in `../v1/`. This ladder is a new generation.
Walk-forward folds select. **One sealed holdout, opened once after H-011**, not after every hyp.
Panel construction is infrastructure: `01_data/data_files/s2_coint/v2/s2_pair_panel.ipynb` (not a hypothesis).
yfinance has no delisted prices: the panel builder reports the drop rate; every hyp quotes it.
Alpaca `shortable` today is not a historical borrow flag.
Kill the generation if holdout net Sharpe < 0.3 or more than 40% of PnL comes from one pair.

Research IS / sealed holdout / `PAPER_START` come from `01_data/data_files/s2_coint/v2/s2_cuts.json` (written in Pass A). Do not reuse v1 letter-universe end dates.

| ID | Question | Status |
|---|---|---|
| H-001 | Pooling (sector vs industry) | OPEN |
| H-002 | Discovery multiple-testing (raw vs BH-FDR) | OPEN |
| H-003 | Bar (1d vs 1h) | OPEN |
| H-004 | Cointegration break | OPEN |
| H-005 | Reshuffle cadence | OPEN |
| H-006 | Book size N | OPEN |
| H-007 | Windows (L / OLS / ADF / entry z / z) | OPEN |
| H-008 | Hedge (OLS vs Kalman) | OPEN |
| H-009 | Overlap | OPEN |
| H-010 | Half-life gate | OPEN |
| H-011 | Vol target | OPEN |

Soft defaults until the relevant STAR exists: freeze book, N=35, `ols=252`, `z=60`, `entry_z=2`, `break_mode=off`, equal 1/N, no HL gate, overlap allow.

**Order is binding:** pooling → FDR → bar → break → cadence → N → windows → hedge → overlap → HL → vol.

Do **not** write STAR notebooks for ADX, ATR/HL exits, corr gate, score sizing, adaptive z, or entry extremity.

---

## H-001 · Pooling

Form pairs only inside the same Yahoo GICS **sector** or **industry** leaf (membership is PIT via `constituents_on_or_before`; labels are current Yahoo GICS). Soft freeze book N=35. Freeze `POOL_STAR` before H-002.

Notebook: `notebooks/hypothesis_tests/H-001_pools.ipynb`.
Next: run H-002.

## H-002 · Discovery multiple-testing

Raw `pvalue < 0.05` versus Benjamini–Hochberg FDR at 5% on the Pass A candidate CSV. **Do not re-run EG.** Freeze `FDR_STAR`.

Notebook: `notebooks/hypothesis_tests/H-002_fdr.ipynb`.
Next: run H-003.

## H-003 · Bar size

1D versus 1H after costs (no 4H). Same close-t / open-t+1 contract. Freeze `BAR_STAR`.

Notebook: `notebooks/hypothesis_tests/H-003_bar.ipynb`.
Next: run H-004.

## H-004 · Cointegration break

`off` | `block_05_flat_10` | `flat_05` on research IS. Freeze `BREAK_STAR` before the book bake-off.

Notebook: `notebooks/hypothesis_tests/H-004_cointegration.ipynb`.
Next: re-run `s2_pair_panel.ipynb` with `BOOK_SCOPE = "freeze_plus_rotate"`.

## H-005 · Reshuffle cadence

`freeze` vs `quarterly` vs `12m6m` (12-month formation / 6-month trading). Soft N=35. Freeze `BOOK_STAR`. Requires Pass B schedules.

Notebook: `notebooks/hypothesis_tests/H-005_cadence.ipynb`.
Next: run H-006.

## H-006 · Book size

N ∈ {20, 35, 50} on the frozen cadence. Freeze `N_STAR`. Slot weight `1/N`.

Notebook: `notebooks/hypothesis_tests/H-006_n.ipynb`.
Next: run H-007.

## H-007 · Windows

Sequential (not Cartesian), after H-005/H-006:

1. Screening `L` ∈ {252, 504} — only if `BOOK_STAR` is `quarterly` or `12m6m`
2. OLS / beta ∈ {252, 504} → `OLS_WINDOW_STAR`
3. ADF ∈ {252, 504} → `ADF_WINDOW_STAR`
4. Entry z ∈ {1.5, 2.0} → `ENTRY_Z_STAR`
5. Z window ∈ {60, 90} → `Z_WINDOW_STAR`

Notebook: `notebooks/hypothesis_tests/H-007_windows.ipynb`.
Next: run H-008.

## H-008 · Hedge

OLS versus Kalman beta. Spread math uses closes only. Freeze `HEDGE_STAR`.

Notebook: `notebooks/hypothesis_tests/H-008_hedge.ipynb`.
Next: run H-009.

## H-009 · Overlap

Allow versus never-allow shared legs. Freeze `OVERLAP_STAR`.

Notebook: `notebooks/hypothesis_tests/H-009_overlap.ipynb`.
Next: run H-010.

## H-010 · Half-life gate

Off versus one pre-registered half-life band. Freeze `HL_GATE_STAR`.

Notebook: `notebooks/hypothesis_tests/H-010_half-life.ipynb`.
Next: run H-011.

## H-011 · Vol target

Book vol target. Choose on research IS only. Do not maximise the holdout. Freeze `VOL_STAR`.

Notebook: `notebooks/hypothesis_tests/H-011_vol.ipynb`.
Next: open the sealed holdout once (v2 OOS tearsheet).
