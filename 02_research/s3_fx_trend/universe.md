# S3 FX Trend — Universe (v1)

A-level locked book for the core and event sleeves. Pair list lives in
`data.ingestion.fx_fetcher.G10_V1_PAIRS` and is mirrored in
`strategies.s3_fx_trend.costs.COSTS["A_FX_OANDA_S3"]["pair_spread_pips"]`.

## Locked pairs (7 vs USD)

| Pair   | Base | Quote | Rationale |
|--------|------|-------|-----------|
| EURUSD | EUR  | USD   | Deepest G10 book; OANDA mid/spread available |
| USDJPY | USD  | JPY   | Core rates / risk-on proxy; JPY pip = 0.01 |
| GBPUSD | GBP  | USD   | Liquid sterling |
| USDCHF | USD  | CHF   | Safe-haven / rate-diff carry |
| USDCAD | USD  | CAD   | Commodity / NA rates |
| AUDUSD | AUD  | USD   | Commodity / Asia risk |
| NZDUSD | NZD  | USD   | Closely related AUD risk; still in confirmed spread table |

All pairs are **quoted against USD** so currency exposure is transparent (USD is
always base or quote) and swap / carry math is `base_rate − quote_rate` without
cross-pair conversion.

Research IS end: **`RESEARCH_IS_END_S3 = 2019-12-31`** (~70/30 on calendar span
from ~2007-01-01 → ~2025-07-11). Sealed OOS is `date > RESEARCH_IS_END_S3`. Same
cut for core and event. Do not re-pick after seeing OOS.

Day boundary: **NY 17:00 America/New_York** (CTA-style FX day). Signal at close
of bar `t`; fill at open of `t+1`.

---

## Dropped: USDSEK / USDNOK

**Not in v1.** Reasons:

1. Not present in the locked OANDA spread table (`A_FX_OANDA_S3`).
2. Typical SEK/NOK retail spreads are wider and more cost-fragile than the seven
   majors above — net Sharpe is dominated by friction before any trend edge.
3. `pair_supported(...)` returns `False` for names outside the table; the
   simulator skips them with a warning rather than inventing a spread.

Revisit only after **confirmed** OANDA spreads / swap for those pairs are
written into `costs.py` and audited in
`02_research/s3_fx_trend/notebooks/data_vendor_tests/03_oanda_costs_confirmed.ipynb`.

---

## Cost model (OANDA S3)

Source of truth: `strategies.s3_fx_trend.costs` (`A_FX_OANDA_S3`).

### Pip size

- Most pairs: **0.0001**
- JPY pairs (`*JPY*`): **0.01**

### Locked half-spread table (full spread in pips)

| Pair   | Spread (pips) |
|--------|---------------|
| EURUSD | 1.4 |
| USDJPY | 1.4 |
| GBPUSD | 2.0 |
| USDCHF | 1.8 |
| USDCAD | 2.2 |
| AUDUSD | 1.4 |
| NZDUSD | 1.7 |

Per-leg execution cost ≈ **½ spread + slippage** in bps of notional.
Default slippage: **0.1 pip per leg**.

### Swap / overnight financing

\[
\text{daily swap return} \approx
\bigl(\underbrace{r_{\text{base}} - r_{\text{quote}}}_{\text{rate differential}}
- \underbrace{50\,\text{bps annual}}_{\text{financing spread}}\bigr)
\times |w| / 365
\]

- Earns the rate differential in the **direction of the signed weight**.
- Financing spread (**50 bps annual**) is always a cost on gross `|w|`.
- **Wednesday triple swap:** OANDA-style triple accrual when `dayofweek == 2`
  (covers weekend roll).
- Policy rates: FRED via `data.ingestion.rates_fetcher` (`fetch_policy_rate` /
  `fetch_all_g10_policy_rates`).

### Conversion markup

**N/A for v1.** All pairs settle vs USD account currency conceptually;
`conversion_markup_bps = 0.0`. Crosses or non-USD account markup would need an
explicit model later.

### Assumptions to revise later

- Spreads are **research locks**, not live quotes — confirm in
  `oanda_costs_confirmed.json` before sealed OOS is decision-grade.
- Slippage is flat (not vol-scaled).
- Swap uses policy-rate proxies, not OANDA’s published financing schedule.
- No weekend gap / holiday liquidity stress beyond Wednesday triple.
- SEK/NOK still out until confirmed.

---

## Related artifacts

| Artifact | Path |
|----------|------|
| Price panel (1D) | `01_data/data_files/s3_fx_trend/s3_price_panel_1d.parquet` |
| Cost table | `05_strategies/s3_fx_trend/costs.py` |
| Cost audit | `04_backtest/s3_fx_trend/artifacts/oanda_costs_confirmed.json` |
| Algorithm notes | `05_strategies/s3_fx_trend/s3_algorithm_notes.md` |
