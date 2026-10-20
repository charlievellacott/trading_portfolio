# S2 v2 hypothesis log — PIT S&P pairs

Version 1 (letter universes, including E) stays in `../v1/`. This ladder is a new generation.
Walk-forward folds select. One sealed holdout, opened once, after the recipe is frozen.
yfinance has no delisted prices: every notebook reports the drop rate. Alpaca `shortable` today is not a historical borrow flag.
Kill the generation if holdout net Sharpe < 0.3 or more than 40% of PnL comes from one pair.

| ID | Question | Status |
|---|---|---|
| H-001 | PIT S&P industry pools | OPEN |
| H-002 | Bar size | OPEN |
| H-003 | Cointegration break | OPEN |
| H-004 | Rolling book | OPEN |
| H-005 | Window screens | OPEN |
| H-006 | Hedge | OPEN |
| H-007 | Trend filter | OPEN |
| H-008 | Half-life gate | OPEN |
| H-009 | Overlap | OPEN |
| H-010 | Exits | OPEN |
| H-011 | Correlation gate | OPEN |
| H-012 | Sizing | OPEN |
| H-013 | Vol target | OPEN |
| H-014 | Z window mode | OPEN |
| H-015 | Entry score | OPEN |

## H-001 · PIT S&P industry pools

Form pairs only inside a point-in-time S&P 500 industry, using constituents_on_or_before and fetch_ohlcv. Report the share of names dropped because yfinance has no delisted prices. Liquidity floor on ADV. Freeze the pool definition before H-002.

Notebook: `notebooks/hypothesis_tests/H-001_pit.ipynb`.

## H-002 · Bar size

1D versus 4H versus 1H after costs, same close-t / open-t+1 contract as v1. Expect to freeze 1D.

Notebook: `notebooks/hypothesis_tests/H-002_bar.ipynb`.

## H-003 · Cointegration break

Same break rule as v1 (block vs flat). Research IS only.

Notebook: `notebooks/hypothesis_tests/H-003_cointegration.ipynb`.

## H-004 · Rolling book

12-month formation and 6-month trading versus a frozen book. Top-N equal risk, N in a small pre-registered set (20, 35, 50). Not the v1 6/2 letter caps.

Notebook: `notebooks/hypothesis_tests/H-004_rolling.ipynb`.

## H-005 · Window screens

Smaller grid than v1: OLS/ADF in {252, 504}, entry z in {1.5, 2.0}, z window in {60, 90}. Fit only after H-004.

Notebook: `notebooks/hypothesis_tests/H-005_window.ipynb`.

## H-006 · Hedge

OLS versus Kalman beta. Spread math uses closes only.

Notebook: `notebooks/hypothesis_tests/H-006_hedge.ipynb`.

## H-007 · Trend filter

Off versus one pre-registered ADX veto. Not a grid.

Notebook: `notebooks/hypothesis_tests/H-007_trend.ipynb`.

## H-008 · Half-life gate

Off versus one pre-registered half-life band.

Notebook: `notebooks/hypothesis_tests/H-008_half-life.ipynb`.

## H-009 · Overlap

Allow versus never-allow shared legs. More important at 20-50 pairs.

Notebook: `notebooks/hypothesis_tests/H-009_overlap.ipynb`.

## H-010 · Exits

No extra exit versus one ATR stop. Half-life time stop stays off unless this arm wins.

Notebook: `notebooks/hypothesis_tests/H-010_exits.ipynb`.

## H-011 · Correlation gate

Off versus one correlation threshold.

Notebook: `notebooks/hypothesis_tests/H-011_correlation.ipynb`.

## H-012 · Sizing

Equal 1/N is the default candidate. Score sizing is the single alternative.

Notebook: `notebooks/hypothesis_tests/H-012_sizing.ipynb`.

## H-013 · Vol target

Book vol target. Choose the target on research IS only. Do not maximise the holdout.

Notebook: `notebooks/hypothesis_tests/H-013_vol.ipynb`.

## H-014 · Z window mode

Fixed window versus one adaptive rule.

Notebook: `notebooks/hypothesis_tests/H-014_z.ipynb`.

## H-015 · Entry score

Traditional z versus one pre-registered extremity score.

Notebook: `notebooks/hypothesis_tests/H-015_entry.ipynb`.
