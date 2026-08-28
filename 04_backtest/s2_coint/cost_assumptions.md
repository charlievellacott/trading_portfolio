# S2 cost assumptions — IBKR Asia (HK / JP)

Source: Interactive Brokers published commissions and third-party fees. Denominations cover HKD / USD / CNH (HK) and JPY (JP).

Note: Borrow rates are completely ignored in this assumption - these are likely the largest costs.

---

## Hong Kong

### IBKR commission (SEHK)


| Monthly trade value (HKD)      | Tiered           | Fixed |
| ------------------------------ | ---------------- | ----- |
| ≤ 15,000,000                   | Tier I — 0.05%   | 0.08% |
| 15,000,000.01 – 300,000,000    | Tier II — 0.05%  | —     |
| 300,000,000.01 – 900,000,000   | Tier III — 0.03% | —     |
| 900,000,000.01 – 2,000,000,000 | Tier IV — 0.02%  | —     |
| > 2,000,000,000                | Tier V — 0.015%  | —     |



| Minimum per order                   | Tiered (I → V)          | Fixed  |
| ----------------------------------- | ----------------------- | ------ |
| SEHK stocks                         | HKD 18 / 12 / 8 / 6 / 4 | HKD 18 |
| SEHK warrants & structured products | HKD 12 / 10 / 8 / 6 / 4 | HKD 10 |


IB SmartRoutingSM applies as on IBKR’s schedule.

### Third-party fees — Hong Kong Stock Exchange (SEHK)


| Fee                               | Rate                                 | Notes                            |
| --------------------------------- | ------------------------------------ | -------------------------------- |
| Exchange                          | 0.00565% of trade value              | All products                     |
| Clearing (exchange trades)        | 0.0042% of trade value               | —                                |
| Clearing (eligible MM ETP trades) | 0.0020% of trade value               | —                                |
| SFC transaction levy              | 0.0027% of trade value               | Normally stocks, warrants, CBBCs |
| FRC transaction levy              | 0.00015% of trade value              | Normally stocks, warrants, CBBCs |
| HK stamp duty                     | 0.1%, rounded up to nearest HKD 1.00 | SEHK stocks only                 |




### Third-party fees — Stock Connect (Northbound: SH–HK / SZ–HK)


| Fee                 | All products            | ETF products           |
| ------------------- | ----------------------- | ---------------------- |
| Handling fee        | 0.00341% of trade value | 0.004% of trade value  |
| Security management | 0.002% of trade value   | Waived                 |
| Transfer fee        | 0.001% of trade value   | Waived                 |
| Stamp duty          | 0.05% of sale proceeds  | 0.05% of sale proceeds |


---



## Japan



### IBKR commission (JP equities)


| Monthly trade value (JPY)           | Tiered           | Fixed |
| ----------------------------------- | ---------------- | ----- |
| ≤ 150,000,000                       | Tier I — 0.05%   | 0.08% |
| 150,000,000.01 – 3,000,000,000      | Tier II — 0.04%  | —     |
| 3,000,000,000.01 – 9,000,000,000    | Tier III — 0.03% | —     |
| 9,000,000,000.01 – 20,000,000,000   | Tier IV — 0.02%  | —     |
| 20,000,000,000.01 – 100,000,000,000 | Tier V — 0.015%  | —     |
| > 100,000,000,000                   | Tier VI — 0.01%  | —     |



| Minimum per order | Tiered (I → VI)                 | Fixed  |
| ----------------- | ------------------------------- | ------ |
| All               | JPY 80 / 70 / 60 / 40 / 30 / 20 | JPY 80 |


IB SmartRoutingSM applies as on IBKR’s schedule. IBKR lists no separate JP regulatory third-party bucket beyond exchange/clearing below.

### Third-party fees — JapanNext Stock Exchange


| Fee                      | Rate                   | IB-JP (incl. tax)       |
| ------------------------ | ---------------------- | ----------------------- |
| Exchange — day session   | 0.2 bps                | 0.22 bps                |
| Exchange — night session | 0.4 bps                | 0.44 bps                |
| Clearing (all products)  | 0.0007% of trade value | 0.00077% of trade value |




### Third-party fees — Tokyo Stock Exchange


| Fee                     | Rate                   | IB-JP (incl. tax)       |
| ----------------------- | ---------------------- | ----------------------- |
| Exchange — Growth & Pro | 0.0056%                | 0.00616%                |
| Exchange — Standard     | 0.0038%                | 0.00418%                |
| Exchange — Prime        | 0.002%                 | 0.0022%                 |
| Clearing (all products) | 0.0007% of trade value | 0.00077% of trade value |


**Note:** Access fees of JPY 2.70 per order apply to all stocks excluding single stocks and basket trading.

---

## US Alpaca — Universe D (realistic profile)

Profile key: `US_ALPACA_D_REALISTIC` in `strategies.s2_coint.costs`. Default for Universe D when loading config from `s2_star_stack.json` (`config_from_stack`).

| Component | Common US leg | Alt share-class (`.A`, `.B`, `NWSA`) |
|-----------|---------------|--------------------------------------|
| Commission | 0 bps | 0 bps |
| Third-party (SEC + FINRA avg) | 0.1 bps | 0.1 bps |
| Slippage (vs daily VWAP) | 3.2 bps | 8.0 bps |
| **Per-leg total (round-trip ×2)** | ~6.6 bps | ~16.2 bps |

| Borrow | Rate | Application |
|--------|------|-------------|
| General collateral short | 100 bps / year | Pro-rated daily on **net short leg weight** while position is open |

**Included vs not (fairness):**

- Included: tiered slippage by share class, flat annual borrow on shorts, calendar-dense Sharpe.
- Not included (stress tier only): vol-scaled slippage, locate / HTB fee spikes, pair-specific bid–ask spread, dividend adjustment on one leg only.

Baseline `US_ALPACA` (flat 3.2 bps/leg, no borrow) remains available via `S2SimConfig(cost_profile=None)` or explicit `COST_PROFILE_STAR` in the stack JSON. **PSR / DSR / Sharpe / max DD** on research tables and tearsheets are computed on **net** returns after the active cost profile (including borrow when applicable).