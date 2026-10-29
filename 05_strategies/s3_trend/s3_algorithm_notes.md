# S3 trend — algorithm notes

- See `strategy_overview.md` and `02_research/s3_trend/universe.md`.

## Universe change (FX book → multi-asset ETFs)

The package was `s3_fx_trend` and the first core freeze was seven USD G10 pairs.
That FX book is now core universe **A** (`FX_G10`). It stays selectable. Core
universe **B** (`ETF_MULTI`) is the equity / bond / commodity / USD ETF book.
H-001 freezes `UNIVERSE_STAR` from a research-IS TSMOM 12M bake-off of A vs B.
`UNIVERSE_STAR` selects the **core** book only.

The **event sleeve is always Forex** (G10 + the economic calendar). It does not
follow `UNIVERSE_STAR`. It will be shelved later; until then it still has an IS
diagnosis and a sealed OOS tearsheet.

Dropped from the active core screen, kept under
`02_research/s3_trend/notebooks/core_sleeve/redundant/`:

- Donchian session filters (old H-007) — ETF daily bars have no FX 17:00 session.
- Carry overlay (old H-010) — FX rate differentials do not map onto the ETF book.
- Value / REER overlay (old H-011) — BIS REER is an FX value measure.

The FX-era core hypothesis log is
`02_research/s3_trend/redundant/s3_core_hypothesis_log_v1_fx.md`.
Active numbering is H-001 … H-011 with no gaps (see the live core log).

## Core universes

| ID | Book | Role |
|---|---|---|
| A | G10 FX vs USD | H-001 candidate |
| B | Multi-asset ETFs | H-001 candidate |

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

- `RESEARCH_IS_END_S3 = 2016-12-31` in `data.processing.s3_price_panel`.
- One cut for FX core, ETF core, and the event sleeve.
- Research IS is `date <= 2016-12-31` (panel from ~2007-01-01).
- Sealed holdout is `2017-01-01` through `2019-12-31`, opened once.
- Dates after `2019-12-31` are a previously seen regime diagnostic, not OOS.
- Do not re-pick the cut after seeing the holdout.

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
S3 wiring: `strategies.s3_trend.metrics`, `backtest.s3_trend.report`.

## Variant ledgers / STAR stacks

| Sleeve | STAR stack | Variant ledger |
|--------|------------|----------------|
| Core | `04_backtest/s3_trend/artifacts/s3_core_star_stack.json` | `s3_variant_ledger_core.json` |
| Event | `04_backtest/s3_trend/artifacts/s3_event_star_stack.json` | `s3_variant_ledger_event.json` |

## Costs

See `universe.md` and `strategies.s3_trend.costs`. `config_from_stack` picks the
profile from `UNIVERSE_STAR` unless `COST_PROFILE_STAR` is set.

- Core A / event sleeve: `A_FX_OANDA_S3` — spread + 0.1 pip slippage per leg;
  swap = rate differential ± **50 bps** financing; **Wednesday triple**.
- Core B: `B_ETF_ALPACA` — 0 commission, 0.1 bps third-party, 3.2 bps open-fill
  slippage (~3.3 bps/leg); ETB borrow 0. Swap off.

## BIS REER (H-011 value) — monthly + PIT M+2

- Source: FRED mirrors of BIS Real Broad REER (`RB*BIS`) via `data.ingestion.alternative_data.bis_reer.fetch_bis_reer`.
- **Monthly only** — no intra-month interpolation of levels.
- PIT availability: month-\(M\) print known on **first calendar day of month\(M+2\)** (conservative). Example: June 2020 usable from 2020-08-01.
- Merge onto daily bars with `merge_asof(..., direction="backward")` on `availability_date`, never on raw month label. No `.bfill` of future months.
- Signal: \(-z(\log\mathrm{REER}_{\mathrm{base}} - \log\mathrm{REER}_{\mathrm{quote}})\); rich base ⇒ lean short base.
- FRED “latest” can revise history; v1 uses lagged latest. **ALFRED vintage is out of scope for v1** (future upgrade).

## Event Calendar and its Viability

- Currently, there is only a frozen source of economic calendar data - which is in the form of a csv.
- If the events sleeve was to pass through the research stage then it would be reasonable to look into a point-in-time and live dataset - however it didn't.
- `clean_calendar_csv` / `load_economic_calendar` prints unmatched `(currency, event_name)` to the console and appends `01_data/data_files/s3_trend/unknown_events.csv` for review. Do not silently drop unknowns.

## Credentials placeholders

Set in `config/credentials.env` (never commit real secrets):

| Variable | Purpose |
|----------|---------|
| `OANDA_API_TOKEN` | OANDA v20 prices (primary) |
| `OANDA_ACCOUNT_ENV` | `practice` or `live` |
| `FRED_API_KEY` | Policy rates + BIS REER |

## Performance

- The algorithm showed very good initial positive Sharpe, in the early years (2008-2012).
- Since then the algorithms Sharpe has agressively fallen, there is significant evidence to show a decay in the Sharpe. 
- Hense by the OOS period the Sharpe had fallen to near 0, and later below 0.
- There was no suggested improvement on the performance with the events sleeve - infact it performed decently worse than the core sleeve.

## Continued development?

- For these reasons above I do not see suitability in keeping this sleeve open, and more importantly deploying it into my portfolio. The sleeve is now closed.