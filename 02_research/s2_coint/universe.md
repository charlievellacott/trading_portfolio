# S2 Universes

Pool definitions live in `01_data/processing/s2_universe_pools.py` (`S2_POOLS`). Pairs are formed
**only within a leaf pool** via `iter_pool_pairs`; the nesting depth is arbitrary (universe →
exchange → sector → tickers) and is walked recursively, so pools can be added or removed by editing
that module alone.

Pair-book construction (frozen at IS end vs quarterly rotating) is decided by **H-004** (`BOOK_STAR`).
Rolling-ADF health gating is decided by **H-003** (`BREAK_STAR`). Discovery Engle-Granger stays out of
the live trading loop in both cases.

A / B / C are kept below as documented learning points. They are **not** hidden and **not** re-selected.

## Results

| Universe | Asset class | Venue | Outcome | Reason |
|----------|-------------|-------|---------|--------|
| A | FX majors | OANDA | Failed | 0 EG passers at IS end |
| B | Crypto | Kraken | Failed | Best pair p=0.0501 and HL > `Z_WINDOW` (60) — untradable horizon |
| C | Asia EM cash equities | IBKR HK / JP | **Shelved** | Gross Sharpe ≈ 0, net negative after 271–439 bps/yr costs; ADF health did not persist (see below) |
| D | US share-class twins | Alpaca | Pending H-001 | — |
| E | US REIT sub-sectors | Alpaca | Pending H-001 | — |
| F | EUR large caps | IBKR EUR | Pending H-001 | — |

Research IS end: A `2020-12-31`, B `2022-12-31`, C `2021-12-31`, **D / E / F `2021-12-31`**.

---

# A: Forex (failed)

- AUDUSD, NZDUSD, EURUSD, GBPUSD, USDCHF, USDJPY (6 names, 15 candidate pairs)

# B: Crypto (failed)

- BTC, ETH, SOL, BNB (4 names, 6 candidate pairs)

# C: (Asian) EM cash equities — shelved

- Universe (11): 0939.HK, 1398.HK, 3988.HK, 1288.HK, 3328.HK, 8306.T, 8316.T, 8411.T, 0857.HK, 0386.HK, 0883.HK

| Yahoo ticker | Actual ticker | Short name  | Sector / what they do     |
| ------------ | ------------- | ----------- | ------------------------- |
| 0939.HK      | 0939          | CCB         | Financial / China bank    |
| 1398.HK      | 1398          | ICBC        | Financial / China bank    |
| 3988.HK      | 3988          | BOC         | Financial / China bank    |
| 1288.HK      | 1288          | ABC         | Financial / China bank    |
| 3328.HK      | 3328          | BoCom       | Financial / China bank    |
| 8306.T       | 8306          | MUFG        | Financial / JP megabank   |
| 8316.T       | 8316          | SMFG        | Financial / JP megabank   |
| 8411.T       | 8411          | Mizuho      | Financial / JP megabank   |
| 0857.HK      | 0857          | PetroChina  | Energy / China oil SOE    |
| 0386.HK      | 0386          | Sinopec     | Energy / China oil SOE    |
| 0883.HK      | 0883          | CNOOC       | Energy / China oil SOE    |

- Leaf pools (no HK↔JP, no bank↔oil): HK China banks (5 names, 10 pairs), JP megabanks (3 names, 3 pairs),
  HK oil SOEs (3 names, 3 pairs). 16 candidate pairs total.
- Locked book was `1398.HK|0939.HK`, `1288.HK|3328.HK`, `8306.T|8316.T`.

## C outcome (shelved)

| Pair | RT/yr | Gross SR | Net SR | Cost bps/yr | RT cost | Median ADF p | % p<0.05 | β std |
|------|-------|----------|--------|-------------|---------|--------------|----------|-------|
| 1288.HK\|3328.HK | ~3.7 | +0.05 | −0.48 | 439 | 116 bps | 0.30 | 11% | 0.45 |
| 1398.HK\|0939.HK | ~3.6 | +0.02 | −0.48 | 422 | 116 bps | 0.19 | 29% | 0.18 |
| 8306.T\|8316.T | ~4.2 | +0.24 | −0.08 | 271 | 64 bps | 0.23 | 18% | 0.24 |

- Not an opportunity problem: `|z|>2` on ~12% of days; ~35–42% of days in-trade.
- Gross Sharpe ≈ 0 on 11–16 years IS → no edge to protect.
- HK friction 116 bps/round-trip vs JP 64 bps → HK net penalty ~2x.
- Spread rarely stationary (median ADF p 0.19–0.30); EG passed once at discovery, health did not persist.
- Pairs not independent: HK bank pairs share one China-bank factor; ADF significance clusters in the same periods.
- Timing contract verified clean (signal close `t` → fill open `t+1`); not an implementation bug.
- Lesson: EG pass at discovery ≠ tradable MR; venue friction must be small vs expected move.

Full postmortem: `02_research/s2_coint/notebooks/other_tests/01_asia_c_failure_diagnosis.ipynb`.
Archived Asia artifacts: `04_backtest/s2_coint/artifacts/asia_c/`.

---

# D: US share-class / structural twins

Leaf pool = one twin set (same issuer, different share class), so the economic tether is a claim on the
same cash flows rather than a statistical coincidence. 10 pools, 20 names, **10 candidate pairs**.

| Pool | Tickers | Company |
|------|---------|---------|
| alphabet | GOOGL, GOOG | Alphabet A / C |
| fox | FOXA, FOX | Fox A / B |
| news_corp | NWSA, NWS | News Corp A / B |
| under_armour | UAA, UA | Under Armour A / C |
| brown_forman | BF.A, BF.B | Brown-Forman A / B |
| lennar | LEN, LEN.B | Lennar A / B |
| heico | HEI, HEI.A | HEICO common / A |
| clearway | CWEN, CWEN.A | Clearway C / A |
| watsco | WSO, WSO.B | Watsco common / B |
| greif | GEF, GEF.B | Greif A / B |

- Per-pool cap of 2 never binds (1 pair per pool); the global cap of 6 does.
- Excluded: BRK.A/BRK.B (share notional), Paramount and Lions Gate (2024 restructurings).
- DISCA/DISCK and RDS.A/RDS.B were genuine twins that **delisted** (WBD merger 2022, Shell
  unification 2022). They are absent from the pools, which is exactly the survivorship gap noted below.

# E: US REIT sub-sectors

Same-subsector REITs share a cap-rate / rates factor and comparable lease economics. 6 pools, 28 names,
**54 candidate pairs**.

| Pool | Tickers |
|------|---------|
| towers | AMT, CCI, SBAC |
| self_storage | PSA, EXR, CUBE, NSA |
| net_lease | O, NNN, ADC, WPC, EPRT |
| industrial | PLD, FR, EGP, STAG, TRNO |
| apartments | AVB, EQR, ESS, MAA, CPT, UDR |
| healthcare | WELL, VTR, OHI, CTRE, SBRA |

- Pools are rate-sensitive as a group, so pools are not mutually independent even though pairs are
  within-pool only. H-004 reports book composition by pool so concentration stays visible.

# F: EUR large caps (same-exchange pools)

7 pools, 26 names, **38 candidate pairs**. Leaf pools are **single-exchange** so trading calendars and
holidays align and the venue-key rule still holds. All names are EUR-denominated, so FX does not enter
the spread itself.

| Pool | Tickers | Exchange |
|------|---------|----------|
| es_banks | SAN.MC, BBVA.MC, CABK.MC, SAB.MC, BKT.MC | Madrid |
| it_banks | ISP.MI, UCG.MI, BAMI.MI, BPE.MI, BMPS.MI | Milan |
| it_regulated_utilities | ENEL.MI, TRN.MI, SRG.MI | Milan |
| es_utilities | IBE.MC, ELE.MC, RED.MC | Madrid |
| nl_semis | ASML.AS, ASM.AS, BESI.AS | Amsterdam |
| de_autos | BMW.DE, MBG.DE, VOW3.DE | Xetra |
| fr_luxury | MC.PA, RMS.PA, KER.PA, OR.PA | Paris |

Risks specific to F:

- **Bank concentration.** Two of seven pools are banks, so with per-pool cap 2 the book can be 4/6 bank
  pairs — the same single-factor concentration that sank C. No theme-level cap is added (avoids knob
  proliferation); H-004 reports composition by pool instead.
- **Short-selling bans.** The Iberian, Italian and French pools sit under regulatory bans in 2011–12 and
  2020, all inside F's research IS. Modelled in `05_strategies/s2_coint/short_bans.py`. `nl_semis`
  (Amsterdam) and `de_autos` (Xetra) are the only F pools never banned — the AFM and BaFin did not
  impose 2020 bans.
- **FX translation.** Sleeve P&L is in EUR; converting to the portfolio base currency adds unhedged FX
  variance. The pair is EUR-neutral only to the extent beta sizing nets the two legs. Not hedged.
- **Cost assumption.** `F_EUR_IBKR` is an assumption pending IBKR's European cash-equity schedule.
  F is the most cost-fragile universe, so this must be confirmed before H-001 scores F.

---

# Known gaps

- **Survivorship bias (documented, not fixed).** Pools are hand-curated from currently listed names, so
  quarterly rotation selects among survivors. Yahoo does not serve delisted tickers, so PIT membership
  is unavailable. D's absent DISCA/DISCK and RDS.A/RDS.B illustrate the size of the gap.
- **Short borrow is not modelled** in any cost profile. Relevant to all of D / E / F.
- **SEC September 2008 US ban** (~799 financial stocks, 2008-09-19 → 2008-10-08) is not modelled for
  D / E. The published list was amended repeatedly and its coverage of equity REITs and share-class
  twins is uncertain; asserting membership would be worse than omitting it. Adding a record later is a
  one-line change to `SHORT_BANS`.
