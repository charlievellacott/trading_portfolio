# S2 cointegration — algorithm notes

- S1 returns are computed weekly and so the correlation was calculated using S2's returns resampled weekly. 
- When `overlap_mode = never_allow` the book simulator makes sure that the days where cash is held flat are still included in the sharpe (such to not over inflate the sharpe by excluding the number of flat days).
- `psr = P(true_SR > 1.0 | T, skew, kurt)` which means probability true Sharpe exceeds 1.0 given sample length T and return shape. Near 1 is good near 0 is bad.
- **DSR (local)** — `dsr_local` = deflated Sharpe after `N_local` arms in **a single** screen/hypothesis. If `dsr_local << ann_sharpe`, the local bake-off may inflate the recorded Sharpe (while only overfitting).
- **DSR (stack)** — Same as local but `N_stack` = cumulative arms in `04_backtest/s2_coint/artifacts/s2_variant_ledger.json`. Penalizes sequential H-001 through current hyp search.
- The **variant ledger** records the permutations for each notebook/hypothesis tested and is used to product the DSR metrics. If a new hypothesis is added this needs to be updated.
- S2 panels fetch **unadjusted** OHLCV (`fetch_ohlcv(..., auto_adjust=False)` in `s2_pair_panel.ipynb`). Adjusted series can phantom-split share-class spreads when one line has dividends and the other does not. β / z / ADF / half-life remain on **close only**; execution still uses opens.
- Research IS is a **fixed calendar end** per universe, registered in `data.processing.s2_universe_pools.RESEARCH_IS_END_BY_UNIVERSE`: A `2020-12-31`, B `2022-12-31`, C `2021-12-31`, D/E/F `2021-12-31`. Screen and `*_train.parquet` both use `date <= T`; sealed OOS is `date > T`. FOR SCREENING ONLY (check for actual).
- Panel columns: `open_y/high_y/low_y/close_y` and `open_x/high_x/low_x/close_x` (no `price_y` / `price_x`). Hedge / z / ADF / half-life use **closes only**; open/high/low are for fill/stop/PnL.
- Uses a screen IS for pairs and freeze (checked a quaterly resample but provided no clear benefit for the risk of much more flase positives).
- **Orientation:** Engle-Granger tests both `y~x` and `x~y` and keeps the lower p-value; the winning direction sets `pair_id`. Orientation is **frozen while a pair is active** and only re-evaluated if the pair is demoted and later re-promoted.
- **Universe C shelved.** Gross Sharpe ≈ 0 (+0.02 to +0.24) before costs, net −0.08 to −0.48 after. Cost drag 271–439 bps/yr (HK 116 bps/RT, JP 64 bps/RT); median rolling ADF p 0.19–0.30, significant only 11–29% of days. Not a timing bug and not a trade-frequency problem (~3.6–4.2 round-trips/yr, `|z|>2` on ~12% of days). Per-pair table in `02_research/s2_coint/universe.md`; archived artifacts in `04_backtest/s2_coint/artifacts/asia_c/`.

## Short-selling bans (mainly EUR)

- Regulator ban windows live in `strategies.s2_coint.short_bans` (`SHORT_BANS`). They block **new entries only** in the spread direction that needs the banned leg: long spread shorts `x`, short spread shorts `y`. Open positions still exit on z, mirroring real bans, which restricted opening or increasing net short positions rather than forcing liquidation.
- Mechanically this is the same mask as demotion, composed by logical AND, via the optional `long_entry_allowed` / `short_entry_allowed` arguments on `engine.simulate_pair`. With both masks absent, behaviour is unchanged.
- Applies to **universe F only** (Madrid / Milan / Paris in 2011–12 and 2020). A–E have no records, so their masks are all-True and their results are byte-identical with or without them. Amsterdam and Xetra were never banned (AFM and BaFin declined in 2020).
- Not modelled: market-maker exemptions, the 2020 net-short thresholds, and the SEC's 2008 US ban (list membership uncertain for REITs and share classes). **Short borrow is still not modelled** in any cost profile.

## H-001 baseline costs / IS diagnostics

- Source of truth: `strategies.s2_coint.costs` (`COSTS`, routing, `leg_cost_bps`). Simulator: `strategies.s2_coint.baseline`. Per-pair IS tables: `strategies.s2_coint.metrics`.
- H-001 notebook (`02_research/s2_coint/notebooks/hypothesis_tests/H-001_universes.ipynb`) calls those modules. Evaluation is **research IS only** (OHLC clipped to `RESEARCH_IS_END` before the pair panel is built).
- Costs are modeled per leg at entry and exit via `COSTS`:
  - `A_FX_OANDA`: spread+slippage model with pair-level spread pips.
  - `B_CRYPTO_KRAKEN`: maker/taker + slippage model (baseline assumes taker).
  - `C_HK_IBKR` / `C_JP_IBKR`: percent commission + minimum + third-party fee + slippage.
  - `US_ALPACA` (universes D / E): **commission-free**, so cost is regulatory fees (SEC Section 31 + FINRA TAF, 0.1 bps) plus 3.2 bps execution → **3.3 bps/leg, ~13 bps per pair round trip**. Execution cost is broker-agnostic slippage against a VWAP benchmark, and the upper end of the published band is used deliberately because a pair fills two legs at once and one is a short. Roughly **9x less friction than C's HK 116 bps/RT**, which is the quantitative case for the universe pivot.
  - `US_ALPACA_D_REALISTIC` (Universe D default via `config_from_stack`): common leg **3.2 bps** slippage; alt share-class lines (`.A`, `.B`, `NWSA`) **8.0 bps**; **100 bps/year** borrow pro-rated daily on net short leg weight. **Not modelled:** vol-scaled slippage, locate fees, HTB spikes, pair-specific spread — document as stress-only fairness bounds in research notes.
  - `F_EUR_IBKR` (universe F): percent commission + EUR minimum + third-party + slippage → ~10.5 bps/leg. **This is an assumption**, not a confirmed schedule: IBKR's published Reg-NMS metrics describe US stocks and the 0.1% / EUR 4 / EUR 29 table is the mutual-fund schedule, so neither applies to EUR cash equities. Confirm Fixed vs Tiered and per-exchange minimums for Madrid / Milan / Amsterdam / Xetra / Paris before treating F's net Sharpe as decision-grade; F is the most cost-fragile universe.
- Market routing is deterministic:
  - `=X` -> `A_FX_OANDA`
  - `-USD` -> `B_CRYPTO_KRAKEN`
  - `.HK` -> `C_HK_IBKR`
  - `.T` -> `C_JP_IBKR`
  - `.MC` / `.MI` / `.AS` / `.DE` / `.PA` -> `F_EUR_IBKR`
  - plain US symbols and US share-class lines (`GOOGL`, `AMT`, `BF.B`) -> `US_ALPACA`
- Baseline sizing uses hedge ratio (`beta`) per bar: long spread `+y, -beta*x`; short spread `-y, +beta*x`. Trad-z exit is a signed recross of `EXIT_Z` (default 0): flatten a long spread when `z >= 0`, a short when `z <= 0`.
- Per-pair IS stats: trade count, median hold (completed round-trips), cost bps/year, Sharpe, max DD, rolling ADF (`adf_pvalue` from `compute_coint_metrics`).
- Asia C IS postmortem → `02_research/s2_coint/notebooks/other_tests/01_asia_c_failure_diagnosis.ipynb` (helpers in `04_backtest/s2_coint/diagnosis.py`).
- **Universe C shelved.** Gross Sharpe ≈ 0 (+0.02 to +0.24) before costs, net −0.08 to −0.48 after. Cost drag 271–439 bps/yr (HK 116 bps/RT, JP 64 bps/RT); median rolling ADF p 0.19–0.30, significant only 11–29% of days. Not a timing bug and not a trade-frequency problem (~3.6–4.2 round-trips/yr, `|z|>2` on ~12% of days). Per-pair table in `02_research/s2_coint/universe.md`; archived artifacts in `04_backtest/s2_coint/artifacts/asia_c/`.
- **Universe D shelved.** WSO.B is not shortable via Alpaca or Interactive Brokers (IBKR). Locked book was `WSO.B|WSO`, `NWS|NWSA`, `HEI|HEI.A`; `WSO|WSO.B` delivered the entirety of the returns. No further broker search. H-001 drops US tickers with Alpaca `shortable=False` before EG screening.
- **Alpaca `shortable` vs `easy_to_borrow`.** Gate is `shortable` (broker will take the short). `easy_to_borrow` is the ETB list; HTB names can be shortable but not ETB. Displayed in H-001, not a gate. Universes in `SKIP_SHORTABILITY_UNIVERSES` (currently `F`) are not queried.
- Overlay check: compound IS daily book returns to S1 Monday–Monday weeks and correlate vs `01_data/data_files/s1_equities/s1_period_returns.parquet` (exported from `08_oos_tearsheet.ipynb`). Missing file → `corr_to_s1` is NaN.

## Live paper (hardcoded STAR)

- Frozen recipe is `04_backtest/s2_coint/artifacts/s2_star_stack.json` (see `01_star_tearsheet.ipynb`). Paper runner: `07_execution/s2_coint/s2_paper_runner.py`. Dedicated Alpaca paper account (100% of that account equity). Credentials: `S2_ALPACA_API_KEY` / `S2_ALPACA_SECRET_KEY` in `config/credentials.env` (same file as S1; **never** falls back to S1 keys). Logs: `07_execution/s2_coint/logs/s2_paper_YYYYMMDD.txt` (not the S1 log dir). Live ledger: `09_performance/cache/live_s2/`. Cache: `05_strategies/s2_coint/cache/` (`S2_CACHE_DIR` override).
- Clock: fill morning of `t+1`. Features from last completed close `t` (drop any `date >= fill_date`). Wait until 09:28 ET, then DAY market deltas (no resting stops; STAR `EXIT_STAR=mean_only`). `--dry-run` prints orders and does not submit.

- **Score sizing denominator (live, not in the STAR tearsheet):** `mean_abs_score` is the per-pair rolling mean of `|z|` with `window = Z_WINDOW_STAR` (frozen **90**, not the H-001 panel default 60) and `min_periods = window`, through close `t`. The sealed backtest (`01_star_tearsheet.ipynb`) instead freezes the **research-IS mean of `|z|`** via `fit_mean_abs_score` and does not refit on OOS. Rolling live scale was **not** walk-forwarded — backtest it before treating live sizing as research-parity.

## Metric explainations
- Sortino ratio = the returns over the standard deviation of the downside. So it does not penalise for upside volatility. A Sortino ratio close to or less than the Sharpe ratio suggests heavy or skewed downside risk (Sortino < Sharpe = downside risk).
- CAGR = Compound Annual Growth Rate. Thus is like the yearly returns.
- CVAR = Conditional Value at Risk --> "If everything goes completely wrong and I hit a catastrophic tail risk event, what is the average amount of money I will lose?"
- Calamar ratio = the CAGR divided by the |max DD|. Calamar > 1 is considered good as your CAGR is greater than your worst drawdown. The ratio does not scale with leverage.
- Win rate = the fraction of days with a positive book return, not % of trades won.
- Profit factor = a measure of gross profit per gross loss. So how much it makes for every dollar lost. Its like the average risk to reward of the strategies trades.
- bps = "basis points" where 1 bps = 0.01% of a given value. EG) In a management fee 1 bps is 0.01% of total captial invested or in a brokers commission it is of the total cash value of the trade. Costs bps per year = the exact percentage of your total portfolio value taken out annually to cover fees.
- Skew = a measure of asymmetry of a return distribution showing whether extreme gains or losses are more likely. If Skew < 0 then long left tail (and thus regular winning but rare total ruin). If Skew > 0 then long right tail (and thus frequent small losses but unlimited upside). A negative skew is bad, a positive skew is good - asuming the **mean** (not median) return is greater than 0. A skew between -0.5 and +0.5 is considered negligable.
> Note: If a high positive skew is accompanied by an extreme excess kurtosis then the tails are likely so fat that the risk of extreme crashes is highly present (typically masked by high positive spikes). This can be accounted for using a probabilistic Sharpe Ratio (which is different to a penalised or adjusted Sharpe it tracks the probability that a strategies true Sharpe ratio is greater than a benchmark). 
- Kurtosis = a measure of fattness or heaviness of tails in a distribution. Excess kurtosis is the kurtosis of a distribution miunus 3 (since a perfect normal distribution has a kurtosis of 3). If excess kurtosis is < 0 then there are unusually thin tails and if excess kurtosis is > 0 tails are fatter. An excess kurtosis > 3 (thus kurtosis > 6) is generally considered highly significant (thus showing that typical risk metrics eg Sharpe are failing). This is because it poses large risk exposure to crashes.
- Half-Kelly is a mathematical formula used to find the optimal fraction of capital to risk or the ideal amount of leverage to maximize long term geometric growth (then as Half-Kelly it cuts it in half). It is calculated as 0.5 x(p - (1-p)/b) where p is the probability of winning (1-p thus prob of losing) and b the average risk-to-reward (aka win-loss ratio). 
- A t-statistic (t-stat) is a measure of magnitude and direction of a signal (not necessarily significance). It measures the number of standard errors your sample data sits away from the null hypothesis. A positive value suggests outperformance while a negative underperformance. A t-stat greater than 2 is generally considered significant.