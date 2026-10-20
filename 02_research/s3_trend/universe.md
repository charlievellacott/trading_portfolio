# S3 trend — universes

`UNIVERSE_STAR` selects the **core** book only. The event sleeve always uses
Forex (universe A prices, OANDA costs, NY 17:00, economic calendar) and ignores
this key.

One research cut for every sleeve:

| Window | Dates |
|---|---|
| Research IS | 2007-01-01 through **2016-12-31** (`RESEARCH_IS_END_S3`) |
| Sealed holdout | 2017-01-01 through 2019-12-31 (open once) |
| Previously seen regime | 2020-01-01 onward (diagnostic only, never a STAR decision) |

The joint start is 2007 because that is when the liquid ETF set and the G10
book are both usable. Earlier single-ETF history is not the book.

## A — FX_G10 (selectable, not the default)

Seven USD pairs: EURUSD, USDJPY, GBPUSD, USDCHF, USDCAD, AUDUSD, NZDUSD.
Costs: OANDA spread, slippage, and swap (`A_FX_OANDA_S3`). Day boundary:
17:00 America/New_York.

USDSEK / USDNOK stay out (not in the confirmed spread table).

This is the v1 core book. Do not refit it. The old hypothesis log is
`redundant/s3_core_hypothesis_log_v1_fx.md`.

## B — ETF_MULTI (default)

| Sleeve | Tickers |
|---|---|
| Equities | SPY, QQQ, IWM, EFA, EEM, EWJ, FXI |
| Bonds and credit | SHY, IEF, TLT, TIP, LQD, HYG |
| Commodities | GLD, SLV, DBC, USO, DBA |
| USD | UUP |

Costs: ETF spread + commission, plus a short-borrow assumption on short legs.
A long-only diagnostic arm is allowed. Signal at the cash close of `t`, fill at
the next session open. There is no FX 17:00 session on this book.

Registry: `data.processing.s3_fx_price_panel.S3_CORE_UNIVERSES`.

## Event sleeve

Always A. Notebooks stay in `notebooks/event_sleeve/`. Shelve later.
