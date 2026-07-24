# Hypothesis Log


| ID    | Date       | Asset    | Factor                  | Data required                                                                   | Status      |
| ----- | ---------- | -------- | ----------------------- | ------------------------------------------------------------------------------- | ----------- |
| H-001 | 2026-07-04 | Equities | OBV-Confirmed Momentum  | Daily OHLCV panel                                                               | PENDING     |
| H-002 | 2026-07-04 | Equities | GK Vol Ratio (Reversal) | Daily OHLC                                                                      | PENDING     |
| H-003 | 2026-07-04 | Equities | Idiosyncratic Vol Rank  | Daily OHLCV panel + SPY daily returns                                           | PENDING     |
| H-004 | 2026-02-07 | Equities | Beta Feature Suite      | Daily OHLCV + SPY + ETF Carhart proxies (Tier A)                                | IMPLEMENTED |
| H-005 | 2026-02-07 | Equities | Size & Value            | Daily OHLCV + SEC Company Facts → daily mcap/P/E/P/B (`fetch_size_value_daily`) | PENDING     |
| H-006 | 2026-07-24 | Equities | 52-Week High Proximity  | Daily OHLCV (`close`, `high`)                                                   | PENDING     |
| H-007 | 2026-07-24 | Equities | MAX (Lottery Demand)    | Daily OHLCV closes → daily returns                                              | PENDING     |
| H-008 | 2026-07-24 | Equities | Gross Profitability     | SEC Company Facts (Revenue, COGS, Assets) + daily panel                         | PENDING     |
| H-009 | 2026-02-07 | Equities | Sentiment               | Alpha Vantage or GDELT; FinBERT for headline scoring                            | PENDING     |
| H-010 | 2026-07-06 | Equities | GBM vs RNN vs Ensemble  | Daily OHLCV (PIT via `fetch_top_n_equities`) + production feature set           | PENDING     |
| H-011 | 2026-02-07 | Equities | Autocorrelation         | —                                                                               | PENDING     |


---



## H-001 · Equities · OBV-Confirmed Momentum · 2026-07-04


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **What it is**         | Price momentum retained only when On-Balance Volume trend agrees with price direction; unconfirmed moves are zeroed or down-weighted vs raw momentum.                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Hypothesis**         | OBV-confirmed momentum has higher next-week (and next-day) predictive power than raw momentum alone.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Economic rationale** | Price moves backed by cumulative volume flow reflect informed participation; price moves without volume support are more likely to fade. Complementary to H-006 (52-week high) as a momentum-family signal.                                                                                                                                                                                                                                                                                                                                              |
| **Data required**      | Daily OHLCV panel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Test to complete**   | Paired IC and quintile spreads on the daily panel: raw momentum vs OBV-confirmed momentum vs forward returns at Alphalens `periods=(1, 5, 21)` (primary narrative 5d; also 1d via price/volume alignment table). Screen window grids on research IS only (see H-010 sample discipline).                                                                                                                                                                                                                                                                   |
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
| **Test to complete**   | Quintile spread and IC of GK vol ratio vs forward returns at Alphalens `periods=(1, 5, 21)` on the daily panel (primary 5d); winsorise ratio and floor denominator in research cleaning; horse-race / nested IC vs H-007 MAX (both short-horizon reversal/lottery family). Screen windows on research IS only (see H-010).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
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
| **Test to complete**   | Quintile spread and IC of idio-vol rank vs forward returns at Alphalens `periods=(1, 5, 21)` on the daily panel (primary 5d); compare to total realised vol rank as baseline; horse-race / nested IC vs H-007 MAX (lottery vs IVOL). Screen on research IS only (see H-010).                                                                                                                                                                                                                                                                               |
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
| **Test to complete**   | Alphalens IC and quintile spreads for all 11 features at `periods=(1, 5, 21)` on the train parquet. Screen window grids `[60, 126, 252]` and residual-momentum `skip=[21, 63]` on research IS only (H-010 sample discipline). Compare to raw beta baseline.                                                                   |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                             |
| **Notes**              | See design notes below.                                                                                                                                                                                                                                                                                                       |


**Design notes**

- **Workspace pattern:** `_ensure_spy_workspace` runs 3 univariate OLS (full/down/up) per ticker per window; `_ensure_ff_workspace` runs 1 multivariate 4-factor OLS (Carhart-style). Both are idempotent — called once for the full window list and results cached as `_ws_`* columns on the panel. Store callers are thin algebra + optional CS-rank.
- `benchmark='spy'|'ff'` **parameter** on `add_beta` and `add_residual_momentum`. FF outputs carry the `smart_` prefix.
- `smart_` **prefix rule:** any column derived from the 4-factor (ETF Carhart proxies + Momentum) regression is prefixed `smart_`.
- **Normalize policy:** `normalize=True` (CS pct-rank) for beta-family and smart-betas (regime-dependent distributions). **No `normalize` kwarg** on `add_blume_beta` or `add_residual_momentum` — those outputs are never CS-ranked (Blume shrinkage / residual-momentum magnitude would be discarded).
- **4-factor merge decision (Carhart):** one 4-factor regression `(r_stock − rf) = α + b₁·MktRF + b₂·SMB + b₃·HML + b₄·Mom + ε` serves both smart-beta slopes AND `smart_residual_mom` (4-factor residuals). Reduces total OLS fits from 5 to 4 per stock per window position.
- **Multi-window screening contract:** all window kwargs accept `int | list[int]`. Passing a list produces the cartesian product of columns with window-suffix naming (matching H-002). `parse_beta_factor_name()` decodes any H-004 column back to its parameters for the IC-loop. Single value → bare name (no suffix).
- `min_obs_conditional = max(20, window // 4)` on conditional β⁻/β⁺.
- **No floor, no winsorize** in library code. Bad inputs → NaN.
- **Excess-return convention:** FF workspace uses `log_return(close) − rf` as the dependent variable.
- **Factor source (active):** ETF Tier A proxies in `data.ingestion.alternative_data.fama_french_fetcher` via `fetch_ohlcv` — `rf=BIL`, `mkt_rf=SPY−rf`, `smb=IWM−SPY`, `hml=IWD−IWF`, `mom=MTUM−SPY`; cache `etf_ff_factors_daily.parquet`.
- **Learning note (Ken French → ETF):** Dartmouth ZIPs are free but monthly-lagged and revised → not PIT for live / train–serve. Archived ZIP fetcher: `02_research/notebooks/redundant/old_fama_french_fetcher.py`; archived notebook: `02_research/notebooks/redundant/old_H-004_beta.ipynb`. Schema kept identical so this is an explicit replacement, not a silent cover-up. **Transferable rule:** freeze only features you can recompute on the decision clock (also applies to sentiment, fundamentals, other vendor archives).
- **References:** Ang, Chen & Xing (2006) "Downside Risk"; Blitz, Huij & Martens (2011) "Residual Momentum".

**Features (11 columns via 8 store callers)**


| #   | Store caller                             | Output column(s)                                    | Normalize |
| --- | ---------------------------------------- | --------------------------------------------------- | --------- |
| 1   | `add_beta(benchmark='spy')`              | `beta` / `beta_{W}`                                 | True (default) |
| 2   | `add_beta(benchmark='ff')`               | `smart_beta_smb/hml/mom` [`_{W}`]                   | True (default) |
| 3   | `add_downside_beta`                      | `downside_beta` [`_{W}`]                            | True (default) |
| 4   | `add_upside_beta`                        | `upside_beta` [`_{W}`]                              | True (default) |
| 5   | `add_net_beta_spread`                    | `net_beta_spread` [`_{W}`]                          | True (default) |
| 6   | `add_relative_downside_beta`             | `rel_downside_beta` [`_{W}`]                        | True (default) |
| 7   | `add_relative_upside_beta`               | `rel_upside_beta` [`_{W}`]                          | True (default) |
| 8   | `add_blume_beta`                         | `blume_beta` [`_{W}`]                               | **none** (never CS-ranked) |
| 9   | `add_residual_momentum(benchmark='spy')` | `residual_mom` / `residual_mom_{K}_{S}`             | **none** (never CS-ranked) |
| 10  | `add_residual_momentum(benchmark='ff')`  | `smart_residual_mom` / `smart_residual_mom_{K}_{S}` | **none** (never CS-ranked) |


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
| **Notes**              | Size is the market cap. **Data source (fetcher):** SEC EDGAR Company Facts + daily OHLCV via `fetch_size_value_daily` in `data.ingestion.alternative_data`. Daily `market_cap` / `pe` / `pb` are reconstructed (filing-dated fundamentals `merge_asof` backward onto closes; PIT on `filed`, not period end). Join for research: `panel.merge(sv, on=["date","ticker"], how="left")`. Default SEC User-Agent: `trading_portfolio charlie.vellacott@gmail.com` (override via kwarg / `SEC_USER_AGENT`). No API key. **Implemented features (store callers in `feature_store.py`):** `add_book_yield` → `book_yield` (1/pb); `add_earnings_yield` → `earnings_yield` (1/pe); `add_log_mcap` → `log_mcap` (log market cap); `add_valuation_roc` → `val_roc_{metric}` (Δlog over window L, metric='pe' or 'pb'); `add_size_momentum` → `size_mom` (log mcap RoC, multi-window via list); `add_value_momentum_interaction` → `val_mom_interact` (cs_rank(by) × cs_rank(mom)); `add_value_momentum_distance` → `val_mom_dist` (sqrt((1-mom_rank)^2 + (1-val_rank)^2), ideal=(1,1)); `add_value_momentum_residual` → `val_mom_resid` (standardised residual from rolling OLS of cs_rank(by) on cs_rank(mom)). **Normalization:** `normalize=True` (CS pct-rank) kept on `add_book_yield`, `add_earnings_yield`, `add_log_mcap` (level features). **No `normalize` kwarg** on `add_valuation_roc`, `add_size_momentum`, `add_value_momentum_interaction`, `add_value_momentum_distance`, `add_value_momentum_residual` — those outputs are never CS-ranked (already Δlog / rank-space / z-scored). **Edge cases:** no floor, no winsorize; pe/pb/mcap ≤ 0 → NaN. **Momentum input:** `raw_momentum` from `momentum.py` (lookback=252, skip=21 default). **Expectations:** Earnings yield / book yield are classic value — expect moderate IC. Valuation RoC ("getting cheaper/richer") often predicts better at 5-21d than static level. Size (log mcap) rank — expect weak/unstable short-horizon IC in large-cap S&P sleeves, but useful as GBM covariate. Size momentum overlaps price momentum — test incremental IC. Value-momentum interaction / distance / residual — lit shows value and momentum are negatively correlated; interaction often lifts IC spreads. **Issues:** SEC data sparse (quarterly filings only); fundamental-dependent features will have stale periods between filings; S&P 500 sleeve is large-cap biased reducing size-effect power. |




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



## H-006 · Equities · 52-Week High Proximity · 2026-07-24


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **What it is**         | Ratio of today’s close to the highest high over the prior `W` trading days (default 252 ≈ 52 weeks). Near 1.0 = trading near its high; near 0 = deep drawdown from the high.                                                                                                                                                                                                                                                                                     |
| **Hypothesis**         | Stocks closer to their 52-week high outperform stocks farther from it over the next 5–21 days (positive IC), after controlling for raw intermediate momentum.                                                                                                                                                                                                                                                                                                   |
| **Economic rationale** | Anchoring / underreaction: investors use the 52-week high as a reference point and underreact to good news for names near the high (George & Hwang 2004). This is a *level-of-path* signal, not a return-over-lookback signal like H-001 or H-004 residual momentum.                                                                                                                                                                                            |
| **Data required**      | Daily OHLCV panel (`close`, `high`).                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Test to complete**   | Alphalens IC + quintiles at `periods=(1, 5, 21)` on `s1_factor_panel_train.parquet`. Screen `W ∈ {126, 252, 504}` on research IS only (H-010 sample discipline). **Baselines:** raw mom (`L=252,S=21`), H-001 OBV-confirmed mom, H-004 residual mom. Keep only if **incremental** mean IC or nested GBM gain vs those baselines.                                                                                                                              |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Notes**              | `normalize=True` → CS pct-rank within date (GBM-ready). Expect positive monotonicity (high ratio → long). Large-cap sleeve OK — not a small-cap liquidity story. Optional later: simple trend/vol overlay (former HMM idea; see Potential ideas) — do **not** treat regime as a CS factor. Multi-window: `W` accepts `int` or list → `near_52w` / `near_52w_{W}`.                                                                                              |


**Formulae**

- Rolling peak: `Hmax_{t,W} = max(high_{t-W+1}, …, high_t)` (document whether today is included; keep one PIT convention)
- Factor (raw): `near_52w = close_t / Hmax_{t,W}`
- Optional transform: `log_drawdown = ln(near_52w)` (more linear in deep drawdowns)
- Stored CS feature: `pct_rank(near_52w)` within date `t`
- Label (primary): `P_{t+5} / P_t - 1` (also 1d, 21d)

**References:** George, T. & Hwang, C.-Y. (2004). “The 52-Week High and Momentum Investing.” *Journal of Finance*.

---



## H-007 · Equities · MAX (Lottery Demand) · 2026-07-24


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **What it is**         | Average of the `N` largest daily simple (or log) returns over the past `W` trading days (Bali et al. default: `N=5`, `W=21`). High MAX = “lottery-like” recent path.                                                                                                                                                                                                                                                                                                      |
| **Hypothesis**         | High MAX predicts **lower** next-week returns (negative IC) after controlling for total realised vol and idiosyncratic vol (H-003).                                                                                                                                                                                                                                                                                                                                      |
| **Economic rationale** | Preferential demand for lottery-like payoffs (Bali, Cakici & Whitelaw 2011): retail / constrained investors overpay for stocks with extreme recent upside days; subsequent returns mean-revert. Related to, but not identical to, the idiosyncratic-volatility puzzle (H-003) and the GK reversal family (H-002).                                                                                                                                                        |
| **Data required**      | Daily OHLCV closes → daily returns.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Test to complete**   | Alphalens IC + quintiles at `(1, 5, 21)` on train parquet. Screen `W ∈ {10, 21, 42}`, `N ∈ {1, 5}` on IS only (H-010). **Baselines:** CS-ranked realised vol; H-003 idio-vol rank; H-002 GK reversal. Report partial / nested IC: MAX alone, MAX ⊥ idio-vol, idio-vol ⊥ MAX. Keep the survivor(s).                                                                                                         |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Notes**              | Sign: high MAX → short / low rank for long book. Store as CS pct-rank of MAX (or of `-MAX` if “higher = better” for GBM). Cost-sensitive at 1d; primary narrative still 5d. Not momentum — MAX uses extreme order statistics, not cumulative return. Column naming: one combo → `max_lottery`; multi → `max_lottery_{N}_{W}`.                                                                            |


**Formulae**

- Daily return: `r_t = P_t / P_{t-1} - 1`
- In window of last `W` returns ending at `t`: average the `N` largest values → `MAX_t`
- Special case `N=1`: `MAX = max(r_{t-W+1}, …, r_t)`
- Factor for ranking: CS pct-rank of `MAX` (expect **negative** IC vs forward return)
- Optional: `MAX_resid` = residual of CS-rank(MAX) on CS-rank(idio_vol) within date (incremental lottery effect)
- Label (primary): `P_{t+5} / P_t - 1`

**References:** Bali, T., Cakici, N. & Whitelaw, R. (2011). “Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns.” *Journal of Financial Economics*.

---



## H-008 · Equities · Gross Profitability · 2026-07-24


| Field                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **What it is**         | Gross profitability = `(Revenue − COGS) / Assets` (or GrossProfit / Assets when the SEC tag exists), held PIT from filing date via `merge_asof` backward onto the daily panel — same Company Facts pattern as H-005.                                                                                                                                                                                                                                            |
| **Hypothesis**         | High gross profitability predicts higher forward returns (positive IC) at 5d/21d, incremental to H-005 book/earnings yield (value) and size.                                                                                                                                                                                                                                                                                                                     |
| **Economic rationale** | Novy-Marx (2013): profitable firms earn higher average returns than unprofitable firms; GP is a cleaner quality measure than net income because it is upstream of SG&A, interest, and taxes. Quality and value are negatively correlated in the cross-section — combining H-005 value with GP often lifts spreads.                                                                                                                                            |
| **Data required**      | SEC EDGAR Company Facts tags for Revenue, COGS (or GrossProfit), and Total Assets; extend `fetch_size_value_daily` / sibling fetcher; daily OHLCV panel for join keys.                                                                                                                                                                                                                                                                                            |
| **Test to complete**   | Alphalens IC + quintiles at `(1, 5, 21)` on train parquet (H-010 discipline). Compare to H-005 `book_yield` / `earnings_yield` / `log_mcap`. Nested IC / GBM gain: GP alone vs GP + value. Screen none or light (levels, not windows) — optional TTM vs latest quarterly.                                                                                                                       |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Notes**              | Sparse between filings (same staleness as H-005). PIT on `filed`, not period end. `normalize=True` CS pct-rank by default. Prefer `us-gaap` GrossProfit when present; else Revenue − CostOfGoodsSold / CostOfRevenue. Assets: `Assets` with documented fallback. Expect stronger IC at 21d than 1d. Do not implement until Revenue/COGS/Assets tags are wired into the SEC fetcher.                                                                           |


**Formulae**

- Gross profit: `GP = GrossProfit` if reported, else `Revenue − COGS`
- Gross profitability: `gp_asset = GP / Assets` (NaN if Assets ≤ 0 or missing)
- Stored feature: CS pct-rank of `gp_asset` within date `t` → column `gross_profitability`
- Label (primary): `P_{t+5} / P_t - 1`

**References:** Novy-Marx, R. (2013). “The Other Side of Value: The Gross Profitability Premium.” *Journal of Financial Economics*.

---



## H-009 · Equities · Sentiment · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                 |
| **Hypothesis**         | Higher sentiment = higher movement in price, etc.                                                                                                                                                                                                       |
| **Economic rationale** | Sentiment is used to explain the percieved value of a stock, thus depending on vibe certain retail investors may back off or go into a specific stock.                                                                                                  |
| **Data required**      | Test Alpha Vantage; if not works historically use GDELT Dataset. Deploy with Alpha V' and apply transformations due to training.                                                                                                                        |
| **Test to complete**   | Firstly, is there a tradable signal? (if applicable: use the past week of news headlines to create a sentiment score using FinBert) Then, explore the decay of the signal (especially if using GDELT as need to decide on a window of articles to use). |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                       |
| **Notes**              | Need to work out how to get a hold of the data first. Vendor probe notebook: `02_research/notebooks/other_tests/H-009_gdelt_data_vendors.ipynb`.                                                                                                        |


---



## H-010 · Equities · GBM vs RNN vs Ensemble · 2026-07-06


| Field                  |                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                                                                                                 |
| **What it is**         | GBM vs RNN vs ensemble. GBM: pooled (single model, all tickers — not one model per ticker). Features: final production set. Run after factor features are built.                                                                                                                                                                                                        |
| **Hypothesis**         | No single architecture dominates all forward horizons; pooled GBM, shared-weight RNN, or an IC-weighted ensemble will win on OOS Alphalens metrics once the production feature set is frozen.                                                                                                                                                                           |
| **Economic rationale** | GBM ranks well on cross-sectional factors; RNN may capture serial structure factors only encode via lags. An ensemble may stabilise errors when the two disagree.                                                                                                                                                                                                       |
| **Data required**      | Daily OHLCV panel via `fetch_top_n_equities` (PIT universe); final production feature pipeline (log feature-spec version in Notes).                                                                                                                                                                                                                                     |
| **Test to complete**   | Walk-forward bake-off: pooled GBM vs shared-weight LSTM/GRU vs validation-IC ensemble on identical splits, labels, and universe. Primary kill/keep weight on **5d** via Alphalens (IC, quantile spreads, turnover); **equal exploration of 21d**; **1d** secondary.                                                                                                     |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                                                                                                       |
| **Notes**              | See configuration, horizon layers, factor-test sample discipline, delisting handling, overfitting controls, and ensemble rule below. H-011 (correlation overlay) is a separate factor — test only after this bake-off. Freeze feature spec before running; note variants tried vs best. Panel OHLCV stays **daily** for all horizons — do not resample to `h`-day bars. |


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



## H-011 · Equities · Autocorrelation · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                            |
| **Hypothesis**         | Multiplying (or the correct linear algebra operation) a prediction matrix (ranking) by a correlation matrix of the stock universe (over a given window) will tell you if the predictions align (or conflict) with each other via the correlations. |
| **Economic rationale** | Autocorrelation                                                                                                                                                                                                                                    |
| **Data required**      | —                                                                                                                                                                                                                                                  |
| **Test to complete**   | —                                                                                                                                                                                                                                                  |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                  |
| **Notes**              | Test idea after creating predictions using the model (H-010), in a notebook or a backtest.                                                                                                                                                         |


---



## Potential ideas

Optional follow-ons if time permits — not numbered hypotheses.

1. **Kalman Filters**
2. **Portfolio management** — hierarchical risk parity, mean-variance optimisation, etc.
3. **GARCH**
4. **Bayesian rolling IC weighting**
5. **Market regime overlay (ex-HMM)** — former H-006 idea: gate or scale positions by a simple trend/vol or HMM state on the index *after* CS scores exist. Improves Sharpe/DD more than mean IC; not a cross-sectional factor.

---



## [Asset Class] — [Factor Name] — [Date proposed]

Hypothesis: (one sentence — why should this predict returns?) 
Economic rationale: (2-3 sentences — what behaviour/structural effect causes this) 
Data required: 
Status: PENDING / KEPT / KILLED 
Test to complete: 
Alphalens summary: (filled in after testing) 
Notes: