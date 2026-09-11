# S3 FX trend — algorithm notes

Companion to [`strategy_overview.md`](strategy_overview.md) and
[`02_research/s3_fx_trend/universe.md`](../../02_research/s3_fx_trend/universe.md).

## Universe (locked)

v1 trades **7 pairs vs USD** only: EURUSD, USDJPY, GBPUSD, USDCHF, USDCAD,
AUDUSD, NZDUSD. **USDSEK and USDNOK are dropped** — not in the confirmed OANDA
spread table; typical SEK/NOK spreads are wide and cost-fragile. Revisit only
after confirmed OANDA costs. Full rationale in `universe.md`.

## Timing contracts

### Core sleeve

- Features / signal / decision at **close of bar `t`** on the NY **17:00**
  America/New_York day boundary.
- Fill at **open of `t+1`**. First PnL from open `t+1` onward.
- Do **not** same-bar close-fill; do **not** add an extra `.shift(1)` on close-`t`
  features.

### Event sleeve

On print time τ, surprise is known at τ. Fill at the **open of the first bar on
`EVENT_BAR_STAR` (`1d` or `1h`) whose open is strictly after τ**. E-003 is the
**daily vs 1h** bake-off; **do not go below 1h**. Core sleeve remains daily
NY 17:00 close → next daily open.

Helper: `data.processing.s3_fx_event_panel.map_event_to_next_bar_fill`.

## Research IS / OOS

- `RESEARCH_IS_END_S3 = 2019-12-31` in `data.processing.s3_fx_price_panel`.
- ~**70/30** split on calendar span (~2007-01-01 → ~2025-07-11).
- Screen / fold-val / full-IS use `date <= RESEARCH_IS_END_S3`.
- Sealed OOS is `date > RESEARCH_IS_END_S3`. Same cut for **core and event**.
- Do not re-pick the cut after seeing OOS.

## Metrics (every hyp notebook)

| Role | Columns |
|------|---------|
| Primary (rank) | `ann_sharpe` (**net**), `max_drawdown` |
| Cost transparency | `ann_sharpe_gross` (pre-cost; compare to net) |
| Co-primary (beside, never combined) | `corr_to_s1`, `corr_to_s2`, `calmar` |
| Secondary | `psr`, `dsr_local`, `dsr_stack`, `skew`, `excess_kurtosis`, `cost_bps_per_year` |
| Context | `n_days`, `n_entries` |

- Rank / STAR hints on **net** Sharpe; report correlations beside it — **no
  combined score**. Negative corr to S1/S2 beats low positive.
- **`psr` = P(true SR > 0)** given sample length and return shape (Bailey &
  López de Prado PSR with benchmark 0). Near 1 is good; near 0 is bad.
- **`dsr_local`** — deflated Sharpe after `N_local` arms in the current screen.
- **`dsr_stack`** — same with `N_stack` = cumulative arms in the sleeve variant
  ledger.
- **Calmar** = CAGR / |max DD|.
- Do **not** put Sortino / win rate / profit factor / half-Kelly / t-stat in
  default tables.

Canonical math: `09_performance/sharpe_inference.py`.
S3 wiring: `strategies.s3_fx_trend.metrics`, `backtest.s3_fx_trend.report`.

## Variant ledgers / STAR stacks

| Sleeve | STAR stack | Variant ledger |
|--------|------------|----------------|
| Core | `04_backtest/s3_fx_trend/artifacts/s3_core_star_stack.json` | `s3_variant_ledger_core.json` |
| Event | `04_backtest/s3_fx_trend/artifacts/s3_event_star_stack.json` | `s3_variant_ledger_event.json` |

Load/save via `backtest.star_stack_io`. Register arms with
`backtest.s3_fx_trend.research.register_hypothesis_arms(..., sleeve=...)`.
STAR assignment is **manual** in notebooks — never auto-assign from
`median_sharpe_hint`.

## Costs

See `universe.md` and `strategies.s3_fx_trend.costs`:

- Spread + 0.1 pip slippage per leg.
- Swap = rate differential ± **50 bps** financing; **Wednesday triple**.
- Conversion markup **N/A** (0 bps) for v1 USD-quoted book.
- Confirm before sealed OOS:
  `04_backtest/s3_fx_trend/artifacts/oanda_costs_confirmed.json`.

## BIS REER (H-011 value) — monthly + PIT M+2

- Source: FRED mirrors of BIS Real Broad REER (`RB*BIS`) via
  `data.ingestion.alternative_data.bis_reer.fetch_bis_reer`.
- **Monthly only** — no intra-month interpolation of levels.
- PIT availability: month-\(M\) print known on **first calendar day of month
  \(M+2\)** (conservative). Example: June 2020 usable from 2020-08-01.
- Merge onto daily bars with `merge_asof(..., direction="backward")` on
  `availability_date`, never on raw month label. No `.bfill` of future months.
- Signal: \(-z(\log\mathrm{REER}_{\mathrm{base}} - \log\mathrm{REER}_{\mathrm{quote}})\);
  rich base ⇒ lean short base.
- FRED “latest” can revise history; v1 uses lagged latest. **ALFRED vintage is
  out of scope for v1** (future upgrade).

## Live Event Calendar Viability

- **Research source:** `01_data/data_files/s3_fx_trend/economic_calendar.csv`
  (~2007 → ~Aug 2025). Calendar `actual` / `forecast` / `previous` are **not
  vintage-PIT**; execution timing (fill after print τ) is still PIT relative to
  the recorded print clock. Event STARs from this file are **research /
  diagnostic only**.
- **Live needs:** feed ≥ chosen event bar; local PIT vault; shared `event_id`
  map; OANDA prices at `EVENT_BAR_STAR`.
- **Available / planned:** static CSV; **`thisweek.json`** (~1h — interim live
  adapter via `load_thisweek_calendar` stub); Finnhub free poll→vault; paid
  TE/Finnhub later.
- **Do not paper-trade events off the CSV alone.**

## Unknown event mappings

`clean_calendar_csv` / `load_economic_calendar` prints unmatched
`(currency, event_name)` to the console and appends
`01_data/data_files/s3_fx_trend/unknown_events.csv` for review. Do not silently
drop unknowns.

## Credentials placeholders

Set in `config/credentials.env` (never commit real secrets):

| Variable | Purpose |
|----------|---------|
| `OANDA_API_TOKEN` | OANDA v20 prices (primary) |
| `OANDA_ACCOUNT_ENV` | `practice` or `live` |
| `FRED_API_KEY` | Policy rates + BIS REER |

Hyp notebooks may fall back to `source="yfinance"` for daily prices when OANDA
is unavailable; FRED-backed panels still need `FRED_API_KEY`.

## Vendor validation

See `02_research/s3_fx_trend/notebooks/data_vendor_tests/`. Flag OANDA vs
yfinance failures here when audits fail.
