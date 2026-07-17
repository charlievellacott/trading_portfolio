# Hypothesis Log


| ID    | Date       | Asset    | Factor                  | Data required                                                         | Status  |
| ----- | ---------- | -------- | ----------------------- | --------------------------------------------------------------------- | ------- |
| H-001 | 2026-07-04 | Equities | OBV-Confirmed Momentum  | Daily OHLCV panel                                                     | PENDING |
| H-002 | 2026-07-04 | Equities | GK Vol Ratio (Reversal) | Daily OHLC                                                            | PENDING |
| H-003 | 2026-07-04 | Equities | Idiosyncratic Vol Rank  | Daily OHLCV panel + SPY daily returns                                 | PENDING |
| H-004 | 2026-02-07 | Equities | Beta                    | Daily stock returns + SPY daily returns (beta)                        | PENDING |
| H-005 | 2026-02-07 | Equities | Size & Value            | Daily OHLCV panel + daily market cap + valuation ratios (P/E, P/B)    | PENDING |
| H-006 | 2026-02-07 | Equities | HMM Trend Regime        | —                                                                     | PENDING |
| H-007 | 2026-02-07 | Equities | Sentiment               | Alpha Vantage or GDELT; FinBERT for headline scoring                  | PENDING |
| H-008 | 2026-07-06 | Equities | GBM vs RNN vs Ensemble  | Daily OHLCV (PIT via `fetch_top_n_equities`) + production feature set | PENDING |
| H-009 | 2026-02-07 | Equities | Autocorrelation         | —                                                                     | PENDING |


---



## H-001 · Equities · OBV-Confirmed Momentum · 2026-07-04


| Field                  |                                                                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                     |
| **What it is**         | Price momentum retained only when On-Balance Volume trend agrees with price direction; unconfirmed moves are zeroed or down-weighted vs raw momentum.                                       |
| **Hypothesis**         | OBV-confirmed momentum has higher next-week (and next-day) predictive power than raw momentum alone.                                                                                        |
| **Economic rationale** | Price moves backed by cumulative volume flow reflect informed participation; price moves without volume support are more likely to fade. Pairs with H-006 regime work (momentum in trends). |
| **Data required**      | Daily OHLCV panel.                                                                                                                                                                          |
| **Test to complete**   | Paired IC and quintile spreads: raw momentum vs OBV-confirmed momentum vs next-week return; repeat for 1d horizon using price/volume alignment table below.                                 |
| **Alphalens summary**  | —                                                                                                                                                                                           |
| **Notes**              | Compare marginal lift over raw momentum on same universe and rebalance schedule. `normalize=True` by default in `add_obv_confirmed_momentum` (cross-sectional pct-rank of the combined signal within each date), so the stored feature is cross-sectional / GBM-ready; set `normalize=False` for the raw combined signal. Soft mode also uses cross-sectional pct-rank of OBV trend as the weight. |


**Formulae**

- Raw momentum: `P_{t-S} / P_{t-L} - 1` (e.g. L = 252 days, S = 21 days skip)
  - **lookback (`L`)**: how far back the start price is (e.g. 252 ≈ 12 months)
  - **skip (`S`)**: how far back the end price is (e.g. 21 ≈ 1 month)
  - Together: return from ~12 months ago to ~1 month ago, skipping the most recent month so short-term reversal does not contaminate the momentum signal
- OBV: add volume on up days, subtract on down days; `OBV_t = OBV_{t-1} + sign(P_t - P_{t-1}) * V_t`
- OBV trend: `OBV_t - OBV_{t-W}` (e.g. W = 20 days)
- Confirmed momentum: keep raw momentum only when its sign matches OBV trend sign; else 0
- Label (next-week return): `P_{t+5} / P_t - 1`

**Price vs volume alignment (suggested next day)**


| Price momentum | OBV trend | Suggested next day        |
| -------------- | --------- | ------------------------- |
| Positive       | Rising    | Long / hold long          |
| Positive       | Falling   | Flat / reduce exposure    |
| Negative       | Falling   | Short / hold short        |
| Negative       | Rising    | Flat / cover (divergence) |


---



## H-002 · Equities · GK Vol Ratio (Reversal) · 2026-07-04


| Field                  |                                                                                                                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                     |
| **What it is**         | Ratio of short-window Garman–Klass (intraday OHLC) volatility to longer-window realised close-to-close volatility; high values flag intraday stress not fully reflected in closes.                                                          |
| **Hypothesis**         | High GK/realised vol ratio predicts negative next-week returns (short-horizon mean reversion).                                                                                                                                              |
| **Economic rationale** | Wide intraday ranges with muted close-to-close vol suggest two-way fighting, liquidity shocks, or intraday overreaction that partially reverses — opposite to momentum. Overlaps H-005 volume-spike reversal idea but uses range-based vol. |
| **Data required**      | Daily OHLC (open, high, low, close).                                                                                                                                                                                                        |
| **Test to complete**   | Quintile spread and IC of GK vol ratio vs 5-day forward return; winsorise ratio and floor denominator; test interaction with H-006 regime (reversal stronger in ranging markets).                                                           |
| **Alphalens summary**  | —                                                                                                                                                                                                                                           |
| **Notes**              | Reversal signals are turnover- and cost-sensitive — report net of spread/slippage. May conflict with H-001 on same names; test as overlay or let GBM combine. `normalize=True` by default in `add_gk_vol_ratio` (cross-sectional pct-rank of the mode-transformed ratio within each date), so the stored feature is CS / GBM-ready; set `normalize=False` for the unranked value. Store does **not** floor the realised-vol denominator (non-positive → NaN ratio) and does **not** winsorize — apply winsorize in research/cleaning if needed. Realised vol = population std of the last `realised_window` log returns ending at `t`. **Std convention:** prefer **`ddof=0` (population)** for rolling realised vol — matches numpy / most vol literature, and with `normalize=True` CS ranks are identical for `ddof=0` vs `ddof=1` (same date, same window). Use **`ddof=1` (sample / Bessel)** only if reconciling against a library that defaults to sample std (e.g. pandas `.std()`). The implementation hardcodes population std inside `realised_vol` (not a store kwarg). With `normalize=True`, CS ranks of `ratio` and `log_ratio` coincide when the ratio is positive (monotonic). |


**Formulae**

- Garman–Klass variance: `0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2`
- GK vol: square root of variance (clip at zero if negative)
- Realised vol: population std of the last W log returns `ln(C_t / C_{t-1})` ending at `t` (default W = 20; needs W+1 closes)
- Ratio: short-window mean of daily GK vol (default 5) divided by realised vol
- Modes (column `gk_vol_{mode}`): `ratio` (raw), `log_ratio` (`ln` of positive ratio), `reversal` (`-` raw ratio)
- Label (next-week return): `P_{t+5} / P_t - 1`

---



## H-003 · Equities · Idiosyncratic Vol Rank · 2026-07-04


| Field                  |                                                                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Status**             | PENDING                                                                                                                                                                                                                                                      |
| **What it is**         | Cross-sectional rank of each stock's 20-day residual return volatility after stripping out market exposure.                                                                                                                                                  |
| **Hypothesis**         | Low idiosyncratic-vol rank (quieter stock-specific noise) predicts higher next-week returns; high rank predicts lower returns (idiosyncratic volatility puzzle).                                                                                             |
| **Economic rationale** | Lottery preference and short-sale constraints leave high idio-vol names overpriced; arbitrageurs more easily correct mispricing in low idio-vol names. Related to H-004 low-vol / BAB literature but isolates stock-specific rather than market-linked risk. |
| **Data required**      | Daily OHLCV panel + SPY daily returns (market benchmark).                                                                                                                                                                                                    |
| **Test to complete**   | Quintile spread and IC of idio-vol rank vs 5-day forward return; compare to total realised vol rank as baseline.                                                                                                                                             |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                            |
| **Notes**              | Benchmark locked to SPY. Expect negative monotonicity (low rank → long). Use PIT universe and cross-sectional rank on each date only. Purge/embargo for overlapping 5d labels.                                                                               |


**Formulae**

- Daily log return: `ln(P_t / P_{t-1})`
- Rolling 20-day OLS: `r_i = alpha + beta * r_SPY + epsilon`
- Idiosyncratic vol: standard deviation of `epsilon` over 20 days
- Factor: cross-sectional percentile rank of idio vol on date t
- Label (next-week return): `P_{t+5} / P_t - 1`

---



## H-004 · Equities · Beta · 2026-02-07


| Field                  |                                                                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                       |
| **Hypothesis**         | The key weakness for a stock picking strategy is when the market is down. My having lower beta stocks the amount of alpha would be increased. |
| **Economic rationale** | When a crash occurs the predicted (GBM) top n stocks (if have low alpha) will still move higher irrespectable of the market.                  |
| **Data required**      | Daily stock returns (universe panel) + SPY daily returns (market benchmark for beta).                                                         |
| **Test to complete**   | What features can I extract from beta?                                                                                                        |
| **Alphalens summary**  | —                                                                                                                                             |
| **Notes**              | CONSIDER: Compare vs low-beta / low-vol anomaly (Frazzini & Pedersen, "Betting Against Beta").                                                |


---



## H-005 · Equities · Size & Value · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                       |
| **Hypothesis**         | 1. Use a "normalized rate of change of valuation (of a stock)"? (size momentum or a market-cap growth factor). 2. Volume spikes during trending markets could indicate a reversal and vice versa for a ranging market.                                                                       |
| **Economic rationale** | Volume spikes just after (or while still in) a highly bullish market indicate high selling pressure. RoC in valuation dictacts the percieved growth of a company.                                                                                                                             |
| **Data required**      | Daily OHLCV panel (volume spikes, price direction); daily market cap per ticker (size / size momentum); valuation ratios (P/E, P/B) for valuation rate-of-change.                                                                                                                             |
| **Test to complete**   | Explore how change in size is effected based on the direction of the stock before. Look at the effects of volume spikes when the market is moving in different directions - compare the volume spike senarios to controls (where there are no volume spikes) but similar movements in price. |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                             |
| **Notes**              | Size is the market cap                                                                                                                                                                                                                                                                        |


---



## H-006 · Equities · HMM Trend Regime · 2026-02-07


| Field                  |                                                                                                                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                             |
| **Hypothesis**         | Surely buying and selling should align with the stocks in a bull and bear market - this focuses on momentum as opposed to mean reversion.                                           |
| **Economic rationale** | Momentum                                                                                                                                                                            |
| **Data required**      | —                                                                                                                                                                                   |
| **Test to complete**   | Look at Hurst exponent or variance ratio tests. No matter where the model ranked stocks if the trade direction does not match the market direction (HMM state) then don't trade it. |
| **Alphalens summary**  | —                                                                                                                                                                                   |
| **Notes**              | Test idea with a backtest and a plot that details the agreement between the final direction, prediction and the HMM state.                                                          |


---



## H-007 · Equities · Sentiment · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                 |
| **Hypothesis**         | Higher sentiment = higher movement in price, etc.                                                                                                                                                                                                       |
| **Economic rationale** | Sentiment is used to explain the percieved value of a stock, thus depending on vibe certain retail investors may back off or go into a specific stock.                                                                                                  |
| **Data required**      | Test Alpha Vantage; if not works historically use GDELT Dataset. Deploy with Alpha V' and apply transformations due to training.                                                                                                                        |
| **Test to complete**   | Firstly, is there a tradable signal? (if applicable: use the past week of news headlines to create a sentiment score using FinBert) Then, explore the decay of the signal (especially if using GDELT as need to decide on a window of articles to use). |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                       |
| **Notes**              | Need to work out how to get a hold of the data first.                                                                                                                                                                                                   |


---



## H-008 · Equities · GBM vs RNN vs Ensemble · 2026-07-06


| Field                  |                                                                                                                                                                                                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                     |
| **What it is**         | GBM vs RNN vs ensemble. GBM: pooled (single model, all tickers — not one model per ticker). Features: final production set. Run after factor features are built.                                                                                            |
| **Hypothesis**         | No single architecture dominates all forward horizons; pooled GBM, shared-weight RNN, or an IC-weighted ensemble will win on OOS Alphalens metrics once the production feature set is frozen.                                                               |
| **Economic rationale** | GBM ranks well on cross-sectional factors; RNN may capture serial structure factors only encode via lags. An ensemble may stabilise errors when the two disagree.                                                                                           |
| **Data required**      | Daily OHLCV panel via `fetch_top_n_equities` (PIT universe); final production feature pipeline (log feature-spec version in Notes).                                                                                                                         |
| **Test to complete**   | Walk-forward bake-off: pooled GBM vs shared-weight LSTM/GRU vs validation-IC ensemble on identical splits, labels, and universe. Primary kill/keep on 5d horizon via Alphalens tearsheet (IC, quantile spreads, turnover). Secondary horizons: 1d, 5d, 21d. |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                           |
| **Notes**              | See configuration, delisting handling, overfitting controls, and ensemble rule below. H-009 (correlation overlay) is a separate factor — test only after this bake-off. Freeze feature spec before running; note variants tried vs best.                    |


**Configuration**

```text
PRIMARY_FORWARD_HORIZON_DAYS: 5   # change only here; purge/embargo and primary Alphalens periods follow this
SECONDARY_FORWARD_HORIZON_DAYS: 1, 10, 21   # exploratory only — do not use for primary kill/keep
```

**Models**

- **GBM (pooled):** one gradient-boosted model trained on all `(date, ticker)` rows stacked together — **not** a separate model per ticker. At inference, pass one row per ticker per day through the same model.
- **RNN:** shared-weight LSTM or GRU — one sequence per ticker, same weights across names. Lookback length tuned on validation only (capped search budget).
- **Ensemble:** compute Spearman IC of GBM and RNN scores vs label on the walk-forward **validation** window only; set `w_i = max(IC_i, 0) / sum(max(IC_j, 0))`; combined score = `w_gbm * score_gbm + w_rnn * score_rnn` (or rank the blend). **Freeze weights before the holdout block** — do not retune on test.

**Label and signal**

- **Primary label:** cross-sectional percentile rank of forward return on date `t` (PIT universe only): rank of `P_{t+h} / P_t - 1` where `h = PRIMARY_FORWARD_HORIZON_DAYS`.
- **Trading signal:** cross-sectional rank of model output at `t` (long top, short bottom). Alphalens factor input = model score or its cross-sectional rank.

**Universe / tensor shape (delistings)**

- Fixed-shape panel `(T, N_max, F)` with **NaN padding** for slots where a ticker is not yet listed or has delisted.
- **Mask** padded positions in RNN loss and in metrics; never forward-fill prices from the future.
- GBM: omit or mask rows with NaN features; do not train on padded ghost tickers.

**Overfitting controls**

- Walk-forward retrain; early stopping on validation only (RNN).
- GBM: `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, limited boosting rounds.
- RNN: dropout, weight decay, gradient clipping; cap hyperparameter trials.
- Winsorize features and labels; fit scalers on train fold only.
- Purged / embargoed CV when forward labels overlap across rows (embargo ≥ primary horizon).
- Same splits, costs, and universe for all three approaches.

**Formulae**

- Forward return: `r_{t,h} = P_{t+h} / P_t - 1`
- Primary label: `label_{i,t} = pct_rank(r_{t,h})` across tradable tickers on date `t`
- Ensemble weight: `w_i = max(IC_i^{val}, 0) / Σ_j max(IC_j^{val}, 0)`

---



## H-009 · Equities · Autocorrelation · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                            |
| **Hypothesis**         | Multiplying (or the correct linear algebra operation) a prediction matrix (ranking) by a correlation matrix of the stock universe (over a given window) will tell you if the predictions align (or conflict) with each other via the correlations. |
| **Economic rationale** | Autocorrelation                                                                                                                                                                                                                                    |
| **Data required**      | —                                                                                                                                                                                                                                                  |
| **Test to complete**   | —                                                                                                                                                                                                                                                  |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                  |
| **Notes**              | Test idea after creating predictions using the model, in a notebook or a backtest.                                                                                                                                                                 |


---



## Potential ideas

Optional follow-ons if time permits — not numbered hypotheses.

1. **Kalman Filters**
2. **Portfolio management** — hierarchical risk parity, mean-variance optimisation, etc.
3. **GARCH**
4. **Bayesian rolling IC weighting**

---



## [Asset Class] — [Factor Name] — [Date proposed]

Hypothesis: (one sentence — why should this predict returns?) 
Economic rationale: (2-3 sentences — what behaviour/structural effect causes this) 
Data required: 
Status: PENDING / KEPT / KILLED 
Test to complete: 
Alphalens summary: (filled in after testing) 
Notes: