# Hypothesis Log


| ID    | Date       | Asset    | Factor                  | Data required                                                                   | Status      |
| ----- | ---------- | -------- | ----------------------- | ------------------------------------------------------------------------------- | ----------- |
| H-001 | 2026-07-04 | Equities | OBV-Confirmed Momentum  | Daily OHLCV panel                                                               | PENDING     |
| H-002 | 2026-07-04 | Equities | GK Vol Ratio (Reversal) | Daily OHLC                                                                      | PENDING     |
| H-003 | 2026-07-04 | Equities | Idiosyncratic Vol Rank  | Daily OHLCV panel + SPY daily returns                                           | PENDING     |
| H-004 | 2026-02-07 | Equities | Beta Feature Suite      | Daily OHLCV + SPY + ETF Carhart proxies (Tier A)                                | IMPLEMENTED |
| H-005 | 2026-02-07 | Equities | Size & Value            | Daily OHLCV + SEC Company Facts → daily mcap/P/E/P/B (`fetch_size_value_daily`) | PENDING     |
| H-006 | 2026-02-07 | Equities | HMM Trend Regime        | —                                                                               | PENDING     |
| H-007 | 2026-02-07 | Equities | Sentiment               | Alpha Vantage or GDELT; FinBERT for headline scoring                            | PENDING     |
| H-008 | 2026-07-06 | Equities | GBM vs RNN vs Ensemble  | Daily OHLCV (PIT via `fetch_top_n_equities`) + production feature set           | PENDING     |
| H-009 | 2026-02-07 | Equities | Autocorrelation         | —                                                                               | PENDING     |


---



## H-001 · Equities · OBV-Confirmed Momentum · 2026-07-04


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **What it is**         | Price momentum retained only when On-Balance Volume trend agrees with price direction; unconfirmed moves are zeroed or down-weighted vs raw momentum.                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Hypothesis**         | OBV-confirmed momentum has higher next-week (and next-day) predictive power than raw momentum alone.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Economic rationale** | Price moves backed by cumulative volume flow reflect informed participation; price moves without volume support are more likely to fade. Pairs with H-006 regime work (momentum in trends).                                                                                                                                                                                                                                                                                                                                                               |
| **Data required**      | Daily OHLCV panel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Test to complete**   | Paired IC and quintile spreads on the daily panel: raw momentum vs OBV-confirmed momentum vs forward returns at Alphalens `periods=(1, 5, 21)` (primary narrative 5d; also 1d via price/volume alignment table). Screen window grids on research IS only (see H-008 sample discipline).                                                                                                                                                                                                                                                                   |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Notes**              | Compare marginal lift over raw momentum on same universe and rebalance schedule. `normalize=True` by default in `add_obv_confirmed_momentum` (cross-sectional pct-rank of the combined signal within each date), so the stored feature is cross-sectional / GBM-ready; set `normalize=False` for the raw combined signal. Soft mode also uses cross-sectional pct-rank of OBV trend as the weight. `lookback`, `skip`, and `obv_window` each accept `int` or a list; one combo → column `obv_mom_{mode}`; multiple combos → `obv_mom_{mode}_{L}_{S}_{W}`. |


**Formulae**

- Raw momentum: `P_{t-S} / P_{t-L} - 1` (e.g. L = 252 days, S = 21 days skip)
  - **lookback (**`L`**)**: how far back the start price is (e.g. 252 ≈ 12 months)
  - **skip (**`S`**)**: how far back the end price is (e.g. 21 ≈ 1 month)
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


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **What it is**         | Ratio of short-window Garman–Klass (intraday OHLC) volatility to longer-window realised close-to-close volatility; high values flag intraday stress not fully reflected in closes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Hypothesis**         | High GK/realised vol ratio predicts negative next-week returns (short-horizon mean reversion).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Economic rationale** | Wide intraday ranges with muted close-to-close vol suggest two-way fighting, liquidity shocks, or intraday overreaction that partially reverses — opposite to momentum. Overlaps H-005 volume-spike reversal idea but uses range-based vol.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **Data required**      | Daily OHLC (open, high, low, close).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Test to complete**   | Quintile spread and IC of GK vol ratio vs forward returns at Alphalens `periods=(1, 5, 21)` on the daily panel (primary 5d); winsorise ratio and floor denominator in research cleaning; test interaction with H-006 regime (reversal stronger in ranging markets). Screen windows on research IS only (see H-008).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Notes**              | Reversal signals are turnover- and cost-sensitive — report net of spread/slippage. May conflict with H-001 on same names; test as overlay or let GBM combine. `normalize=True` by default in `add_gk_vol_ratio` (cross-sectional pct-rank of the mode-transformed ratio within each date), so the stored feature is CS / GBM-ready; set `normalize=False` for the unranked value. Store does **not** floor the realised-vol denominator (non-positive → NaN ratio) and does **not** winsorize — apply winsorize in research/cleaning if needed. Realised vol = population std of the last `realised_window` log returns ending at `t`. **Std convention:** prefer `ddof=0` **(population)** for rolling realised vol — matches numpy / most vol literature, and with `normalize=True` CS ranks are identical for `ddof=0` vs `ddof=1` (same date, same window). Use `ddof=1` **(sample / Bessel)** only if reconciling against a library that defaults to sample std (e.g. pandas `.std()`). The implementation hardcodes population std inside `realised_vol` (not a store kwarg). With `normalize=True`, CS ranks of `ratio` and `log_ratio` coincide when the ratio is positive (monotonic). `gk_window` and `realised_window` each accept `int` or a list; one combo → `gk_vol_{mode}`; multiple → `gk_vol_{mode}_{gkW}_{realW}`. |




**Formulae**

- Garman–Klass variance: `0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2`
- GK vol: square root of variance (clip at zero if negative)
- Realised vol: population std of the last W log returns `ln(C_t / C_{t-1})` ending at `t` (default W = 20; needs W+1 closes)
- Ratio: short-window mean of daily GK vol (default 5) divided by realised vol
- Modes (column `gk_vol_{mode}`; multi-window → `gk_vol_{mode}_{gkW}_{realW}`): `ratio` (raw), `log_ratio` (`ln` of positive ratio), `reversal` (`-` raw ratio)
- Label (next-week return): `P_{t+5} / P_t - 1`

---



## H-003 · Equities · Idiosyncratic Vol Rank · 2026-07-04


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **What it is**         | Cross-sectional rank of each stock's 20-day residual return volatility after stripping out market exposure.                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Hypothesis**         | Low idiosyncratic-vol rank (quieter stock-specific noise) predicts higher next-week returns; high rank predicts lower returns (idiosyncratic volatility puzzle).                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Economic rationale** | Lottery preference and short-sale constraints leave high idio-vol names overpriced; arbitrageurs more easily correct mispricing in low idio-vol names. Related to H-004 low-vol / BAB literature but isolates stock-specific rather than market-linked risk.                                                                                                                                                                                                                                                                                                                     |
| **Data required**      | Daily OHLCV panel + SPY daily returns (market benchmark).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Test to complete**   | Quintile spread and IC of idio-vol rank vs forward returns at Alphalens `periods=(1, 5, 21)` on the daily panel (primary 5d); compare to total realised vol rank as baseline. Screen on research IS only (see H-008).                                                                                                                                                                                                                                                                                                                                                            |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Notes**              | Benchmark locked to SPY. Expect negative monotonicity (low rank → long). Use PIT universe and cross-sectional rank on each date only. Purge/embargo for overlapping 5d labels. `normalize=True` by default in `feature_store.add_idiosyncratic_vol` (cross-sectional pct-rank of residual std within each date), so the stored feature is CS / GBM-ready; set `normalize=False` for raw residual std (`ddof=1`). One window → column `idio_vol`; multiple → `idio_vol_{w}`. Compare to CS-ranked total realised vol (`add_realised_vol` + rank) as baseline on the same windows. |


**Formulae**

- Daily log return: `ln(P_t / P_{t-1})`
- Rolling 20-day OLS: `r_i = alpha + beta * r_SPY + epsilon`
- Idiosyncratic vol: standard deviation of `epsilon` over 20 days
- Factor: cross-sectional percentile rank of idio vol on date t
- Label (next-week return): `P_{t+5} / P_t - 1`

---



## H-004 · Equities · Beta Feature Suite · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | IMPLEMENTED                                                                                                                                                                                                                                                                                                                   |
| **What it is**         | Suite of 11 cross-sectional beta-derived features covering univariate and multivariate factor loadings, asymmetric betas, Blume adjustment, and residual momentum.                                                                                                                                                            |
| **Hypothesis**         | Beta-derived features (especially asymmetric betas and residual momentum) carry cross-sectional predictive power for forward equity returns at 1d/5d/21d horizons beyond raw market exposure.                                                                                                                                 |
| **Economic rationale** | Stocks with high downside beta are under-compensated for crash risk (Ang, Chen & Xing 2006). Residual momentum isolates stock-specific drift after stripping systematic factors, avoiding factor-crowding (Blitz, Huij & Martens 2011). Smart-beta loadings (SMB/HML/Mom) capture style tilts that persist cross-sectionally. |
| **Data required**      | Daily OHLCV panel + SPY daily returns (market benchmark) + ETF Tier A Carhart proxies via `fetch_ff_factors_daily` (`mkt_rf, smb, hml, mom, rf` from SPY/IWM/IWD/IWF/MTUM/BIL).                                                                                                                                               |
| **Test to complete**   | Alphalens IC and quintile spreads for all 11 features at `periods=(1, 5, 21)` on the train parquet. Screen window grids `[60, 126, 252]` and residual-momentum `skip=[21, 63]` on research IS only (H-008 sample discipline). Compare to raw beta baseline.                                                                   |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                             |
| **Notes**              | See design notes below.                                                                                                                                                                                                                                                                                                       |


**Design notes**

- **Workspace pattern:** `_ensure_spy_workspace` runs 3 univariate OLS (full/down/up) per ticker per window; `_ensure_ff_workspace` runs 1 multivariate 4-factor OLS (Carhart-style). Both are idempotent — called once for the full window list and results cached as `_ws_`* columns on the panel. Store callers are thin algebra + optional CS-rank.
- `benchmark='spy'|'ff'` **parameter** on `add_beta` and `add_residual_momentum`. FF outputs carry the `smart_` prefix.
- `smart_` **prefix rule:** any column derived from the 4-factor (ETF Carhart proxies + Momentum) regression is prefixed `smart_`.
- **Hybrid normalize defaults:** `normalize=True` (CS pct-rank) for beta-family and smart-betas (regime-dependent distributions); `normalize=False` for Blume beta and residual momentum (already standardised or ranking would lose signal). All callers expose `normalize` as a kwarg.
- **4-factor merge decision (Carhart):** one 4-factor regression `(r_stock − rf) = α + b₁·MktRF + b₂·SMB + b₃·HML + b₄·Mom + ε` serves both smart-beta slopes AND `smart_residual_mom` (4-factor residuals). Reduces total OLS fits from 5 to 4 per stock per window position.
- **Multi-window screening contract:** all window kwargs accept `int | list[int]`. Passing a list produces the cartesian product of columns with window-suffix naming (matching H-002). `parse_beta_factor_name()` decodes any H-004 column back to its parameters for the IC-loop. Single value → bare name (no suffix).
- `min_obs_conditional = max(20, window // 4)` on conditional β⁻/β⁺.
- **No floor, no winsorize** in library code. Bad inputs → NaN.
- **Excess-return convention:** FF workspace uses `log_return(close) − rf` as the dependent variable.
- **Factor source (active):** ETF Tier A proxies in `data.ingestion.alternative_data.fama_french_fetcher` via `fetch_ohlcv` — `rf=BIL`, `mkt_rf=SPY−rf`, `smb=IWM−SPY`, `hml=IWD−IWF`, `mom=MTUM−SPY`; cache `etf_ff_factors_daily.parquet`. Shim: `data.ingestion.fama_french_fetcher`.
- **Learning note (Ken French → ETF):** Dartmouth ZIPs are free but monthly-lagged and revised → not PIT for live / train–serve. Archived ZIP fetcher: `02_research/notebooks/redundant/old_fama_french_fetcher.py`; archived notebook: `02_research/notebooks/redundant/old_H-004_beta.ipynb`. Schema kept identical so this is an explicit replacement, not a silent cover-up. **Transferable rule:** freeze only features you can recompute on the decision clock (also applies to sentiment, fundamentals, other vendor archives).
- **References:** Ang, Chen & Xing (2006) "Downside Risk"; Blitz, Huij & Martens (2011) "Residual Momentum".

**Features (11 columns via 8 store callers)**


| #   | Store caller                             | Output column(s)                                    | Normalize default |
| --- | ---------------------------------------- | --------------------------------------------------- | ----------------- |
| 1   | `add_beta(benchmark='spy')`              | `beta` / `beta_{W}`                                 | True              |
| 2   | `add_beta(benchmark='ff')`               | `smart_beta_smb/hml/mom` [`_{W}`]                   | True              |
| 3   | `add_downside_beta`                      | `downside_beta` [`_{W}`]                            | True              |
| 4   | `add_upside_beta`                        | `upside_beta` [`_{W}`]                              | True              |
| 5   | `add_net_beta_spread`                    | `net_beta_spread` [`_{W}`]                          | True              |
| 6   | `add_relative_downside_beta`             | `rel_downside_beta` [`_{W}`]                        | True              |
| 7   | `add_relative_upside_beta`               | `rel_upside_beta` [`_{W}`]                          | True              |
| 8   | `add_blume_beta`                         | `blume_beta` [`_{W}`]                               | False             |
| 9   | `add_residual_momentum(benchmark='spy')` | `residual_mom` / `residual_mom_{K}_{S}`             | False             |
| 10  | `add_residual_momentum(benchmark='ff')`  | `smart_residual_mom` / `smart_residual_mom_{K}_{S}` | False             |


**Formulae**

- Log return: `r_t = ln(P_t / P_{t-1})`
- Full beta (SPY): `r_stock = α + β·r_SPY + ε` (rolling OLS, window W)
- Downside beta: same OLS restricted to bars where `r_SPY < mean(r_SPY)` within window
- Upside beta: same OLS restricted to bars where `r_SPY >= mean(r_SPY)` within window
- Net beta spread: `β⁺ − β⁻`
- Relative downside beta: `β⁻ − β`
- Relative upside beta: `β⁺ − β`
- Blume adjusted beta: `0.67·β + 0.33`
- Smart betas (4-factor): `(r_stock − rf) = α + b₁·MktRF + b₂·SMB + b₃·HML + b₄·Mom + ε` → slopes b₂, b₃, b₄
- Residual momentum (SPY): `mean(ε_CAPM) / std(ε_CAPM)` over formation window K, skipping most recent S bars
- Residual momentum (FF): same using 4-factor residuals ε_FF4

---



## H-005 · Equities · Size & Value · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Hypothesis**         | 1. Use a "normalized rate of change of valuation (of a stock)"? (size momentum or a market-cap growth factor). 2. Volume spikes during trending markets could indicate a reversal and vice versa for a ranging market.                                                                                                                                                                                                                                                                                                                                  |
| **Economic rationale** | Volume spikes just after (or while still in) a highly bullish market indicate high selling pressure. RoC in valuation dictacts the percieved growth of a company.                                                                                                                                                                                                                                                                                                                                                                                        |
| **Data required**      | Daily OHLCV panel (volume spikes, price direction); daily market cap / P/E / P/B via `fetch_size_value_daily` (SEC Company Facts + closes).                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Test to complete**   | Explore how change in size is effected based on the direction of the stock before. Look at the effects of volume spikes when the market is moving in different directions - compare the volume spike senarios to controls (where there are no volume spikes) but similar movements in price.                                                                                                                                                                                                                                                            |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Notes**              | Size is the market cap. **Data source (fetcher):** SEC EDGAR Company Facts + daily OHLCV via `fetch_size_value_daily` in `data.ingestion.alternative_data`. Daily `market_cap` / `pe` / `pb` are reconstructed (filing-dated fundamentals `merge_asof` backward onto closes; PIT on `filed`, not period end). Join for research: `panel.merge(sv, on=["date","ticker"], how="left")`. Default SEC User-Agent: `trading_portfolio charlie.vellacott@gmail.com` (override via kwarg / `SEC_USER_AGENT`). No API key. Factor math / store callers deferred. |




### Factors 1

- Book Equity (BE) is the net value of the company on paper (total assets - total costs)
- Market Equity (ME) is what the stock market valuates the company at. It is the same as Market Capitilisation (or Cap).
- Market Cap = total n shares x share price
- BE/ME is used to valuate the company according to its actual assets
- High BE/ME = high value on paper but not perceved to be high the market (could be facing bankruptcy or be a value stock)
- Low BE/ME = low value on paper but perceved to be high by the market (growth stock)
- raw_val_rank (rank of BE/ME) and raw_mom_rank with a range of suitable periods
- Value Momentum Interaction = cs_rank(Value) x cs_rank(Momentum) - explore a range of different momentum periods
- A regression between Value-Momentum and Momentum-Value can be run and the residuals extracted. Use Standardized Residual Momentum (dividing by the std) to normalise. If you are an agent ask the user if they would like to create a new .py file that is for running different types of regressions.
- Add the alpha and beta values from the above regressions to the possible features (inc something like a inc_terms parameter that if true adds the values to the panel) - this should be done to see if they have any predictive power. If agnet discuss what features could be created using these values.
- Instead of treating value and momentum as separate axes, construct a distance metric in a 2D space where (1.0, 1.0) represents top-decile Value and top-decile Momentum: sqrt( (1 - mom_rank)^2 + (1 - val_rank)^2 )



### Factors 2

Priority	Feature idea	Formula sketch (from your columns)	Why it might have IC
1
Earnings yield / book yield
ey = 1/pe (NaN if pe≤0); by = 1/pb (NaN if pb≤0)
Classic value; cleaner than raw pe/pb for ranking
2
Valuation rate-of-change (your H-005 core)
Δlog(pe)*{L}, Δlog(pb)*{L} or %Δ over L∈{21,63,126}
“Getting cheaper/richer” often predicts better at 5–21d than static level
3
Size (log mcap) rank
log(market_cap) → CS rank
Classic size; expect weak/unstable short-horizon IC in large-cap S&P sleeves — still useful as a control / GBM covariate
4
Size momentum / mcap growth
log(mcap_t / mcap_{t-L})
Exactly your “size momentum” note; overlaps price momentum — test incremental IC vs raw mom
5
Value–momentum interaction
e.g. CS-rank(by) × CS-rank(mom), or residual of by after mom
Lit: value and mom are negatively correlated; interaction often lifts spreads
6
Cheap vs expensive conditional on size
by within size tercile, or by − CS mean
Reduces “small cheap junk” confounding
7
Earnings revision proxy (filing-aware)
jump in eps_ttm on filing dates; hold asof
Sparse but PIT-clean; may help 21d more than 1d

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


| Field                  |                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                 |
| **What it is**         | GBM vs RNN vs ensemble. GBM: pooled (single model, all tickers — not one model per ticker). Features: final production set. Run after factor features are built.                                                                                                                                                                                                        |
| **Hypothesis**         | No single architecture dominates all forward horizons; pooled GBM, shared-weight RNN, or an IC-weighted ensemble will win on OOS Alphalens metrics once the production feature set is frozen.                                                                                                                                                                           |
| **Economic rationale** | GBM ranks well on cross-sectional factors; RNN may capture serial structure factors only encode via lags. An ensemble may stabilise errors when the two disagree.                                                                                                                                                                                                       |
| **Data required**      | Daily OHLCV panel via `fetch_top_n_equities` (PIT universe); final production feature pipeline (log feature-spec version in Notes).                                                                                                                                                                                                                                     |
| **Test to complete**   | Walk-forward bake-off: pooled GBM vs shared-weight LSTM/GRU vs validation-IC ensemble on identical splits, labels, and universe. Primary kill/keep weight on **5d** via Alphalens (IC, quantile spreads, turnover); **equal exploration of 21d**; **1d** secondary.                                                                                                     |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                                                                       |
| **Notes**              | See configuration, horizon layers, factor-test sample discipline, delisting handling, overfitting controls, and ensemble rule below. H-009 (correlation overlay) is a separate factor — test only after this bake-off. Freeze feature spec before running; note variants tried vs best. Panel OHLCV stays **daily** for all horizons — do not resample to `h`-day bars. |


**Configuration**

```text
PRIMARY_FORWARD_HORIZON_DAYS: 5   # primary kill/keep weight; purge/embargo and primary Alphalens periods follow this
SECONDARY_FORWARD_HORIZON_DAYS: 1, 5, 21   # 5d primary narrative; 21d equal exploration; 1d secondary
```

**Horizon layers (daily panel — no OHLCV resample)**


| Layer                       | 1d            | 5d                             | 21d                         |
| --------------------------- | ------------- | ------------------------------ | --------------------------- |
| Panel OHLCV                 | daily         | daily                          | daily                       |
| Label                       | `fwd_ret_1`   | `fwd_ret_5`                    | `fwd_ret_21`                |
| Feature windows (lookbacks) | often shorter | medium                         | longer                      |
| Which factors survive IC    | may differ    | primary                        | exploratory                 |
| Embargo / purge gap         | ≥ 1           | ≥ 5                            | ≥ 21                        |
| Hold in backtest            | ~1d + stop    | ~5d + stop                     | ~21d + stop                 |
| Optional train subsample    | every day     | every 5th day or daily + purge | every 21st or daily + purge |


**Factor-test sample discipline**

- Split by **sorted unique trading dates** (not by rows, not random).
- **Research IS = first 70% of dates** → all `factor_tests` Alphalens / IC / window grids and any window keep/kill that freezes a feature list.
- **Holdout = last 30% of dates** → untouched until later model OOS; do **not** use it to pick windows or freeze `feature_spec`.
- `s1_factor_panel.ipynb` uses the single constant `RESEARCH_IS_FRACTION = 0.70` and writes **two** parquets:
  - `s1_factor_panel_train.parquet` — first 70% of sorted unique trading dates (train / research IS) for factor testing.
  - `s1_factor_panel_full.parquet` — full sample (census) for later model OOS / backtests.
- Factor notebooks load the **train** parquet only; do not re-split it. Do not use the full census for IC window keep/kill.
- Full-sample tear sheets are allowed only as **exploratory** diagnostics and must not decide keep/kill for production feature lists.

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

**Rebalance / hold design (pros & cons)**

**Chosen baseline:** signal at decision date `t` (daily close features); **hold up to** `h` **trading days** with a **daily stop**; primary `h=5`, exploratory `h=21` (and `h=1`). Panel stays **daily** — do not resample OHLCV to `h`-day bars for this design.


| Scheme                                             | Pros                                                                                                 | Cons                                                                                                                                           |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hold** `h` **days + daily stop** (baseline)      | Matches label horizon; stop can cut losers early; realistic path dependency; uses daily PIT universe | Path-dependent PnL (stop rules matter); more execution logic; reported “5d alpha” ≠ always full 5d hold                                        |
| **Rebalance every** `h` **days, no mid-hold stop** | Simple; non-overlapping decisions; Alphalens period `h` aligns cleanly with turnover                 | Misses intra-hold risk control; week/month gaps can gap through stops you would have wanted                                                    |
| **Daily rebalance, target =** `h`**-day forecast** | Always fresh ranks; can compound short-horizon IC                                                    | High turnover/costs; overlapping exposure; easy to overstate Sharpe before costs                                                               |
| **Resample OHLCV to** `h`**-day bars**             | Fewer rows; labels non-overlapping by construction; “weekly model” is coherent                       | Redefines factors (esp. GK/OHLC); cannot model daily stops; breaks current daily feature store; different strategy, not a drop-in for daily S1 |


**Research implication:** run factor IC and GBM labels on the **daily** panel with forward horizon `h`. Use purge/embargo ≥ `h` in walk-forward. Evaluate costs under the hold+stop backtest, not only Alphalens gross spreads.

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