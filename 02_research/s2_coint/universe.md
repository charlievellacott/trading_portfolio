# A: (4-6 assets then keep 2-4 pairs for deployment) Forex
- AUDUSD, NZDUSD, EURUSD, GBPUSD, USDCHF, USDJPY

# B: (3-5 assets then keep 2-3 pairs) Crypto
- BTC, ETH, SOL, BNB

# C: (8-15 assets then 3-6 pairs max) (Asian) EM cash equities

## C prior (H-001 first screen)

- Universe (12): 0700.HK, 9988.HK, 3690.HK, 1810.HK, 0939.HK, 1398.HK, 8306.T, 8316.T, 8035.T, 6857.T, 6146.T, 7735.T

| Yahoo ticker | Actual ticker | Short name     | Sector / what they do    |
| ------------ | ------------- | -------------- | ------------------------ |
| 0700.HK      | 0700          | Tencent        | Tech / social & games    |
| 9988.HK      | 9988          | Alibaba        | Tech / e-commerce        |
| 3690.HK      | 3690          | Meituan        | Tech / local services    |
| 1810.HK      | 1810          | Xiaomi         | Tech / devices & IoT     |
| 0939.HK      | 0939          | CCB            | Financial / China bank   |
| 1398.HK      | 1398          | ICBC           | Financial / China bank   |
| 8306.T       | 8306          | MUFG           | Financial / JP megabank  |
| 8316.T       | 8316          | SMFG           | Financial / JP megabank  |
| 8035.T       | 8035          | Tokyo Electron | Semi / wafer equipment   |
| 6857.T       | 6857          | Advantest      | Semi / test equipment    |
| 6146.T       | 6146          | Disco          | Semi / dicing tools      |
| 7735.T       | 7735          | Screen         | Semi / process equipment |

- Sector-specific pair mappings (test **only** these; no HK↔JP, no cross-sector):
  - Tech HK: 0700–9988, 0700–3690, 0700–1810, 9988–3690, 9988–1810, 3690–1810
  - Financial HK banks: 0939–1398
  - Financial JP banks: 8306–8316
  - Semiconductor JP: 8035–6857, 8035–6146, 8035–7735, 6857–6146, 6857–7735, 6146–7735

## C refined (frozen — H-001 DECIDED)

Tech/semis failed IS EG; expand the same-country twins that passed and add China oil SOEs as a disjoint sector cluster.

**Locked book (do not re-screen):** `1398.HK|0939.HK`, `1288.HK|3328.HK`, `8306.T|8316.T`.

H-002 is **1D vs 1H only** for this universe. There is **no 4H** panel or bake-off for C.

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

- Sector-specific pair mappings (test **only** these; no HK↔JP, no bank↔oil):
  - Financial HK banks: all pairs among 0939, 1398, 3988, 1288, 3328 (10)
  - Financial JP banks: 8306–8316, 8306–8411, 8316–8411 (3)
  - Energy HK oil SOEs: 0857–0386, 0857–0883, 0386–0883 (3)
- Note: for production drop the number of **pairs** to 2-3 for most of the universes
- Do not allow for live dynamic pair mining - once pairs have been found keep them.

# Dates for universes
Universe 1 (forex):
  AUDUSD=X      earliest=2006-06-01  recent_ok=True  ok=True
  NZDUSD=X      earliest=2003-12-01  recent_ok=True  ok=True
  EURUSD=X      earliest=2003-12-01  recent_ok=True  ok=True
  GBPUSD=X      earliest=2003-12-01  recent_ok=True  ok=True
  USDCHF=X      earliest=2003-10-01  recent_ok=True  ok=True
  USDJPY=X      earliest=1996-10-01  recent_ok=True  ok=True

Universe 2 (crypto):
  BTC-USD       earliest=2014-09-01  recent_ok=True  ok=True
  ETH-USD       earliest=2017-11-01  recent_ok=True  ok=True
  SOL-USD       earliest=2020-04-01  recent_ok=True  ok=True
  BNB-USD       earliest=2017-11-01  recent_ok=True  ok=True

Universe 3 (asia, C prior):
  0700.HK       earliest=2004-06-01  recent_ok=True  ok=True
  9988.HK       earliest=2019-11-01  recent_ok=True  ok=True
  3690.HK       earliest=2018-09-01  recent_ok=True  ok=True
  1810.HK       earliest=2018-07-01  recent_ok=True  ok=True
  0939.HK       earliest=2005-10-01  recent_ok=True  ok=True
  1398.HK       earliest=2006-10-01  recent_ok=True  ok=True
  8306.T        earliest=2005-09-01  recent_ok=True  ok=True
  8316.T        earliest=2000-01-01  recent_ok=True  ok=True
  8035.T        earliest=2000-01-01  recent_ok=True  ok=True
  6857.T        earliest=2000-01-01  recent_ok=True  ok=True
  6146.T        earliest=2001-01-01  recent_ok=True  ok=True
  7735.T        earliest=2001-01-01  recent_ok=True  ok=True

Universe 3 (asia, C refined — fetch in H-001 prints live ranges):
  0939.HK, 1398.HK, 3988.HK, 1288.HK, 3328.HK, 8306.T, 8316.T, 8411.T, 0857.HK, 0386.HK, 0883.HK

# Cointegration testing vs live health check
- **Discovery (research):** fixed calendar IS end per universe — A `2020-12-31`, B `2022-12-31`, C `2021-12-31`. Screen each candidate on `date <= T`; require at least `min(ols_window, 252)` mutual IS bars (else ineligible, not locked). Confirm locked pairs on holdout after `T` (no all-vs-all re-search).
- **Live / backtest trading:** rolling ADF (or similar) as a kill-switch — pause/flatten when the relationship dies; optional gate on new entries.
- **Do not:** pick pairs on full-sample including holdout; use rolling ADF to invent new pairs in production; treat discovery and live ADF as the same step.
SUMMARY: 22/22 ok

# Test results
- Forex = failed
- Crypto = failed (even though one pair was p=0.0501 its HL was > 60 days and by default the Z-windows are 60)
- Emerging = passed (running more refined)