"""S2 candidate pools: nested ticker groups. Pairs form only within a leaf list.

Structure is intentionally free-form in depth. Every mapping level is a label (universe,
exchange, sector, ...) and every **list of strings is a leaf pool**. ``iter_pool_pairs`` in
``data.processing.s2_universe`` walks any depth recursively, so a pool can be added, removed,
or re-nested here without touching the traversal code.

Canonical dotted ticker form is used throughout (``BF.B``, ``SAN.MC``). The fetcher maps
canonical to Yahoo (``BF.B`` -> ``BF-B``; suffixed names such as ``SAN.MC`` keep the dot when
``isAsian=True``).

Research IS end: A 2020-12-31, B 2022-12-31, C 2021-12-31, D/E/F 2021-12-31.
"""

from __future__ import annotations

# Nested pools keyed by universe letter. Leaf lists are pools; pairs never cross a leaf.
S2_POOLS: dict[str, object] = {
    # ------------------------------------------------------------------
    # A: FX majors (OANDA). Failed H-001 - 0 EG passers. Kept for the record.
    # Single leaf pool: all majors are same-venue and mutually comparable.
    # ------------------------------------------------------------------
    "A": {
        "fx_majors": [
            "AUDUSD=X",
            "NZDUSD=X",
            "EURUSD=X",
            "GBPUSD=X",
            "USDCHF=X",
            "USDJPY=X",
        ],
    },
    # ------------------------------------------------------------------
    # B: Crypto majors (Kraken). Failed H-001 - best pair p=0.0501 with
    # half-life > Z_WINDOW (60), i.e. no reversion inside its own z window.
    # ------------------------------------------------------------------
    "B": {
        "crypto_majors": [
            "BTC-USD",
            "ETH-USD",
            "SOL-USD",
            "BNB-USD",
        ],
    },
    # ------------------------------------------------------------------
    # C: Asia EM cash equities (IBKR HK / JP). SHELVED - gross Sharpe ~ 0,
    # net negative after 271-439 bps/yr, ADF health did not persist.
    # Level 1 = exchange (no HK<->JP pairs), level 2 = sector (no bank<->oil).
    # ------------------------------------------------------------------
    "C": {
        # Hong Kong (.HK)
        "HK": {
            # China "big five" state banks - one shared China-bank factor.
            "cn_banks": ["0939.HK", "1398.HK", "3988.HK", "1288.HK", "3328.HK"],
            # China integrated oil / gas SOEs.
            "cn_oil_soe": ["0857.HK", "0386.HK", "0883.HK"],
        },
        # Tokyo (.T)
        "T": {
            # Japanese megabanks.
            "jp_megabanks": ["8306.T", "8316.T", "8411.T"],
        },
    },
    # ------------------------------------------------------------------
    # D: US share-class / structural twins (Alpaca, commission-free).
    # Leaf pool = one issuer's share classes, so both legs are claims on the
    # same cash flows. Strongest economic tether available; spreads are small,
    # which is why a low-friction venue matters.
    # Per-pool cap of 2 never binds here (one pair per pool); global cap 6 does.
    # Excluded: BRK.A/BRK.B (share notional), Paramount and Lions Gate (2024
    # restructurings). DISCA/DISCK and RDS.A/RDS.B delisted - see survivorship
    # note in 02_research/s2_coint/universe.md.
    # ------------------------------------------------------------------
    "D": {
        "alphabet": ["GOOGL", "GOOG"],  # Class A (voting) / Class C (non-voting)
        "fox": ["FOXA", "FOX"],  # Class A / Class B
        "news_corp": ["NWSA", "NWS"],  # Class A / Class B
        "under_armour": ["UAA", "UA"],  # Class A / Class C
        "brown_forman": ["BF.A", "BF.B"],  # Class A / Class B
        "lennar": ["LEN", "LEN.B"],  # Class A / Class B
        "heico": ["HEI", "HEI.A"],  # common / Class A
        "clearway": ["CWEN", "CWEN.A"],  # Class C / Class A
        "watsco": ["WSO", "WSO.B"],  # common / Class B
        "greif": ["GEF", "GEF.B"],  # Class A / Class B
    },
    # ------------------------------------------------------------------
    # E: US REIT sub-sectors (Alpaca). Leaf pool = property type, so both legs
    # share a cap-rate / rates factor and comparable lease economics.
    # Note: pools are NOT mutually independent (all rate-sensitive); H-004
    # reports book composition by pool so concentration stays visible.
    # ------------------------------------------------------------------
    "E": {
        # Cell towers - long-dated leases, few operators.
        "towers": ["AMT", "CCI", "SBAC"],
        # Self-storage - short leases, same consumer demand driver.
        "self_storage": ["PSA", "EXR", "CUBE", "NSA"],
        # Net lease - bond-like triple-net cash flows, most rate-sensitive.
        "net_lease": ["O", "NNN", "ADC", "WPC", "EPRT"],
        # Industrial / logistics warehouses.
        "industrial": ["PLD", "FR", "EGP", "STAG", "TRNO"],
        # Apartments - classic REIT cointegration group (same rental fundamentals).
        "apartments": ["AVB", "EQR", "ESS", "MAA", "CPT", "UDR"],
        # Healthcare - senior housing, skilled nursing, medical office.
        "healthcare": ["WELL", "VTR", "OHI", "CTRE", "SBRA"],
    },
    # ------------------------------------------------------------------
    # F: EUR large caps (IBKR EUR). Level 1 = exchange, level 2 = sector.
    # Leaf pools are SINGLE-EXCHANGE on purpose: shared trading calendar and
    # holidays keep the pair panel aligned, and every name is EUR-denominated
    # so FX does not enter the spread itself.
    # Short-selling bans (2011-12, 2020) hit the Madrid / Milan / Paris pools -
    # see 05_strategies/s2_coint/short_bans.py. Amsterdam and Xetra were never
    # banned (AFM and BaFin declined in 2020).
    # ------------------------------------------------------------------
    "F": {
        # Madrid (.MC)
        "MC": {
            # Spanish banks - CNMV ban 2011-12, blanket ban 2012-13 and 2020.
            "es_banks": ["SAN.MC", "BBVA.MC", "CABK.MC", "SAB.MC", "BKT.MC"],
            # Spanish utilities / regulated grid.
            "es_utilities": ["IBE.MC", "ELE.MC", "RED.MC"],
        },
        # Milan (.MI)
        "MI": {
            # Italian banks - CONSOB ban 2011-12 and 2020.
            "it_banks": ["ISP.MI", "UCG.MI", "BAMI.MI", "BPE.MI", "BMPS.MI"],
            # Italian regulated utilities / grid operators (tariff-driven).
            "it_regulated_utilities": ["ENEL.MI", "TRN.MI", "SRG.MI"],
        },
        # Amsterdam (.AS) - never subject to a short ban.
        "AS": {
            # Dutch semicap equipment - same lithography / WFE capex cycle.
            "nl_semis": ["ASML.AS", "ASM.AS", "BESI.AS"],
        },
        # Xetra (.DE) - never subject to a short ban.
        "DE": {
            # German autos - same global auto cycle and EU emissions regime.
            "de_autos": ["BMW.DE", "MBG.DE", "VOW3.DE"],
        },
        # Paris (.PA)
        "PA": {
            # French luxury - AMF blanket ban 2020 only.
            "fr_luxury": ["MC.PA", "RMS.PA", "KER.PA", "OR.PA"],
        },
    },
}

# Universes shelved or failed at H-001. Kept in S2_POOLS as documented learning
# points; never re-selected as UNIVERSE_STAR.
SHELVED_UNIVERSES: frozenset[str] = frozenset({"A", "B", "C"})

# Pre-registered research IS end per universe (screen + train split).
RESEARCH_IS_END_BY_UNIVERSE: dict[str, str] = {
    "A": "2020-12-31",
    "B": "2022-12-31",
    "C": "2021-12-31",
    "D": "2021-12-31",
    "E": "2021-12-31",
    "F": "2021-12-31",
}
