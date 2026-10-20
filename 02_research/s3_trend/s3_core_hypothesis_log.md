# S3 trend — core hypothesis log (ETF default)

The FX-era log is `redundant/s3_core_hypothesis_log_v1_fx.md`. Dropped screens
live in `notebooks/core_sleeve/redundant/` (Donchian sessions, carry, value).
This file is the active ladder. Numbering has no gaps.

`UNIVERSE_STAR` is frozen in H-001. Later notebooks read it and do not re-pick.

Research IS ends **2016-12-31**. Sealed holdout is 2017–2019, once. PSR/DSR use
the per-period Sharpe.

| ID | Was | Question | Status |
|---|---|---|---|
| H-001 | new | Core universe A vs B | OPEN — freeze B |
| H-002 | H-001 | TSMOM clears ETF costs | OPEN |
| H-003 | H-002 | Horizon blend | OPEN |
| H-004 | H-003 | Instrument vol target | OPEN |
| H-005 | H-004 | Book vol target | OPEN |
| H-006 | H-005 | Causal spike delever | OPEN |
| H-007 | H-006 | Trend family | OPEN |
| H-008 | H-008 | Conviction scaling | OPEN |
| H-009 | H-009 | Asset-class caps and corr | OPEN |
| H-010 | H-012 | Overlay polish | OPEN — overlays off |
| H-011 | H-013 | Rebalance bands | OPEN |

## H-001 · Universe

Arms: `A` (FX_G10) and `B` (ETF_MULTI). Freeze `UNIVERSE_STAR=B`. A is recorded,
not re-optimised. Notebook: `H-001_universe.ipynb`.

## H-002 · Existence

Single-horizon daily TSMOM must clear ETF costs on the frozen universe
(median fold-val net Sharpe > 0) or the sleeve stops.

## H-003 · Horizon

Arms: `1M`, `3M`, `12M`, `equal_blend`. Freeze `HORIZON_STAR`.

## H-004 · Instrument vol target

Per-instrument vol target versus raw ±1. Freeze `PAIR_VT_STAR`.

## H-005 · Book vol target

Book vol target on top of the instrument target. Freeze `BOOK_VT_STAR`.
The target vol is chosen on research IS only.

## H-006 · Spike

Delever after a vol spike, applied on the **next** bar. Same-day scaling is invalid.

## H-007 · Family

Arms: TSMOM, Donchian, both. No session-definition arm (that notebook is redundant).

## H-008 · Conviction

Sign-only versus scaling by signal strength.

## H-009 · Exposure

Asset-class and sector gross caps, plus a correlation conflict rule.
This replaces the FX currency cap. Arms stay small and pre-registered.

## H-010 · Overlay polish

Carry and value are not in the active screen. This notebook only polishes an
overlay if one was frozen earlier. Expected freeze: overlays off.

## H-011 · Rebalance bands

No-trade band and weak-signal flat, after the trunk is frozen.

## Kill

Stop the ETF core if the sealed holdout net Sharpe is ≤ 0, or if correlation
to SPY in down-markets is above 0.3.
