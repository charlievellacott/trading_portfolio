"""S2 nested pools, ranked book selection, rotation masks, short bans and cost routing."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.s2_coint.rotation import (
    build_active_schedule,
    quarter_end_rebalance_dates,
    schedule_to_masks,
    simulate_frozen_book,
)
from data.processing.s2_coint_store import screen_pair_cointegration
from data.processing.s2_universe import (
    iter_leaf_pools,
    iter_pool_pairs,
    load_s2_pools,
    pool_of_pair,
    pool_path_for_pair,
    pool_tickers,
)
from strategies.s2_coint.book import (
    CAP_GLOBAL,
    CAP_PER_POOL,
    SLOT_WEIGHT,
    BookState,
    apply_rebalance,
    select_book,
    slot_key,
)
from strategies.s2_coint.costs import leg_cost_bps, market_profile_for_ticker
from strategies.s2_coint.engine import simulate_pair
from strategies.s2_coint.short_bans import (
    SHORT_BANS,
    pair_entry_masks,
    short_banned,
)

# Universes with no regulatory short-ban records: masks must be a no-op for these.
_UNBANNED_UNIVERSES = ("A", "B", "C", "D", "E")


# --------------------------------------------------------------------------------------
# Nested pools
# --------------------------------------------------------------------------------------


def test_iter_pool_pairs_depth_two_three_and_four():
    """Any nesting depth works; pairs never cross a leaf pool."""
    depth2 = {"p1": ["AAA", "BBB"], "p2": ["CCC", "DDD"]}
    depth3 = {"lvl": depth2}
    depth4 = {"outer": depth3}
    expected = [("AAA", "BBB"), ("CCC", "DDD")]
    for nested in (depth2, depth3, depth4):
        assert iter_pool_pairs(nested) == expected

    # Unlabelled list-of-lists nesting is also walked.
    assert iter_pool_pairs([["AAA", "BBB"], ["CCC", "DDD"]]) == expected


def test_iter_pool_pairs_no_cross_pool_no_self_and_sorted_legs():
    nested = {"p1": ["BBB", "AAA", "CCC"], "p2": ["DDD", "EEE"]}
    pairs = iter_pool_pairs(nested)
    # Within-pool combinations only.
    assert set(pairs) == {
        ("AAA", "BBB"),
        ("AAA", "CCC"),
        ("BBB", "CCC"),
        ("DDD", "EEE"),
    }
    # No cross-pool pair.
    assert not any({a, b} & {"DDD", "EEE"} and {a, b} & {"AAA", "BBB", "CCC"} for a, b in pairs)
    # No self-pair, legs sorted.
    assert all(a != b and a < b for a, b in pairs)


def test_iter_pool_pairs_rejects_mixed_venue_leaf():
    with pytest.raises(ValueError, match="mixes venues"):
        iter_pool_pairs({"bad": ["0939.HK", "8306.T"]})


def test_leaf_pool_names_are_dotted_paths():
    names = dict(iter_leaf_pools(load_s2_pools("F")))
    assert "MC.es_banks" in names
    assert "AS.nl_semis" in names
    assert pool_path_for_pair("SAN.MC|BBVA.MC", load_s2_pools("F")) == "MC.es_banks"
    # Orientation-insensitive.
    assert pool_path_for_pair("BBVA.MC|SAN.MC", load_s2_pools("F")) == "MC.es_banks"


def test_universe_candidate_counts_match_registered_pools():
    counts = {
        label: len(iter_pool_pairs(load_s2_pools(label))) for label in "ABCDEF"
    }
    assert counts == {"A": 15, "B": 6, "C": 16, "D": 10, "E": 54, "F": 38}
    assert sum(counts.values()) == 139


def test_universe_d_twins_are_pairable_within_pool():
    pairs = iter_pool_pairs(load_s2_pools("D"))
    assert ("GOOG", "GOOGL") in pairs
    assert ("BF.A", "BF.B") in pairs
    # One pair per twin pool, so per-pool cap never binds for D.
    assert len(pairs) == len(pool_of_pair(load_s2_pools("D"))) // 2


# --------------------------------------------------------------------------------------
# Ranked book selection and caps
# --------------------------------------------------------------------------------------


def _screen(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pair_id": pid, "pvalue": p, "eligible": True}
            for pid, p in rows
        ]
    )


def test_select_book_applies_per_pool_then_global_cap():
    screen = _screen(
        [
            ("A1|A2", 0.001),
            ("A1|A3", 0.002),
            ("A2|A3", 0.003),  # third in pool a -> blocked by per-pool cap
            ("B1|B2", 0.004),
            ("B1|B3", 0.005),
            ("C1|C2", 0.006),
            ("D1|D2", 0.007),
            ("E1|E2", 0.008),  # 7th overall -> blocked by global cap
        ]
    )
    pools = {
        "A1|A2": "a", "A1|A3": "a", "A2|A3": "a",
        "B1|B2": "b", "B1|B3": "b",
        "C1|C2": "c", "D1|D2": "d", "E1|E2": "e",
    }
    kept = select_book(screen, pool_of_pair=pools)
    assert len(kept) == CAP_GLOBAL
    assert kept["pool"].value_counts().max() <= CAP_PER_POOL
    assert "A2|A3" not in set(kept["pair_id"])  # per-pool cap
    assert "E1|E2" not in set(kept["pair_id"])  # global cap
    # Ranked best p-value first.
    assert list(kept["pvalue"]) == sorted(kept["pvalue"])
    assert list(kept["rank"]) == list(range(1, len(kept) + 1))


def test_select_book_drops_alpha_failures_and_ineligible():
    screen = pd.DataFrame(
        [
            {"pair_id": "A1|A2", "pvalue": 0.01, "eligible": True},
            {"pair_id": "B1|B2", "pvalue": 0.20, "eligible": True},  # alpha fail
            {"pair_id": "C1|C2", "pvalue": np.nan, "eligible": False},  # too few bars
        ]
    )
    pools = {"A1|A2": "a", "B1|B2": "b", "C1|C2": "c"}
    kept = select_book(screen, pool_of_pair=pools, pvalue_threshold=0.05)
    assert list(kept["pair_id"]) == ["A1|A2"]


def test_select_book_counts_one_slot_per_unordered_pair():
    """A flipped duplicate must not consume a second slot."""
    screen = _screen([("A1|A2", 0.001), ("A2|A1", 0.002)])
    pools = {"A1|A2": "a", "A2|A1": "a"}
    kept = select_book(screen, pool_of_pair=pools)
    assert len(kept) == 1


def test_slot_key_is_orientation_insensitive():
    assert slot_key("A|B") == slot_key("B|A")


# --------------------------------------------------------------------------------------
# Demotion and orientation flips
# --------------------------------------------------------------------------------------


def test_demoted_pair_is_blocked_but_not_dropped_when_open():
    state = BookState()
    apply_rebalance(state, ["A1|A2", "B1|B2"])
    assert state.is_entry_allowed("A1|A2")

    moves = apply_rebalance(state, ["B1|B2"], open_pairs={"A1|A2"})
    assert moves["demoted"] == ["A1|A2"]
    # Blocked from new entries, but still tracked so it can exit on z.
    assert not state.is_entry_allowed("A1|A2")
    assert "A1|A2" in state.blocked


def test_flat_demoted_pair_is_dropped_without_blocking():
    state = BookState()
    apply_rebalance(state, ["A1|A2"])
    moves = apply_rebalance(state, [], open_pairs=set())
    assert moves["demoted"] == ["A1|A2"]
    assert state.blocked == set()


def test_orientation_flip_waits_for_old_side_to_go_flat():
    state = BookState()
    apply_rebalance(state, ["A1|A2"])

    # EG flips direction while a position is open.
    moves = apply_rebalance(state, ["A2|A1"], open_pairs={"A1|A2"})
    assert moves["flipped"] == ["A1|A2->A2|A1"]
    # Neither orientation may open: old is blocked, new is pending.
    assert not state.is_entry_allowed("A1|A2")
    assert not state.is_entry_allowed("A2|A1")
    assert "A1|A2" in state.blocked
    # One slot consumed throughout.
    assert len(state.active) + len(state.pending) == 1

    # Once the old orientation is flat, the flipped one becomes tradable.
    state.release_flat("A1|A2")
    assert state.is_entry_allowed("A2|A1")
    assert len(state.active) == 1


def test_orientation_flip_is_immediate_when_flat():
    state = BookState()
    apply_rebalance(state, ["A1|A2"])
    apply_rebalance(state, ["A2|A1"], open_pairs=set())
    assert state.is_entry_allowed("A2|A1")


# --------------------------------------------------------------------------------------
# Engine entry masks
# --------------------------------------------------------------------------------------


def _pair_panel(n: int = 24, z: np.ndarray | None = None) -> pd.DataFrame:
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    if z is None:
        z = np.zeros(n, dtype=float)
        z[1] = -2.5  # long-spread entry signal
        z[5] = 0.5  # mean recross -> exit
        z[9] = 2.5  # short-spread entry signal
        z[13] = -0.5  # exit
    px = np.linspace(10.0, 12.0, n)
    return pd.DataFrame(
        {
            "date": idx,
            "pair_id": "1398.HK|0939.HK",
            "ticker_y": "1398.HK",
            "ticker_x": "0939.HK",
            "open_y": px,
            "high_y": px,
            "low_y": px,
            "close_y": px,
            "open_x": px * 0.5,
            "high_x": px * 0.5,
            "low_x": px * 0.5,
            "close_x": px * 0.5,
            "alpha": 0.0,
            "beta": 2.0,
            "spread": 0.0,
            "z": z,
            "half_life": 20.0,
            "adf_pvalue": 0.01,
        }
    )


def test_absent_masks_leave_behaviour_identical():
    panel = _pair_panel()
    base = simulate_pair(panel)
    allowed = simulate_pair(
        panel,
        long_entry_allowed=np.ones(len(panel), dtype=bool),
        short_entry_allowed=np.ones(len(panel), dtype=bool),
    )
    assert base.n_entries == allowed.n_entries
    pd.testing.assert_series_equal(base.returns, allowed.returns)


def test_entry_mask_blocks_new_entries_only():
    panel = _pair_panel()
    blocked = np.zeros(len(panel), dtype=bool)
    res = simulate_pair(
        panel, long_entry_allowed=blocked, short_entry_allowed=blocked
    )
    assert res.n_entries == 0


def test_entry_mask_lets_open_position_exit_normally():
    """Block from bar 3 onward: the bar-1 entry must still close on the z recross."""
    panel = _pair_panel()
    mask = np.ones(len(panel), dtype=bool)
    mask[3:] = False
    res = simulate_pair(panel, long_entry_allowed=mask, short_entry_allowed=mask)
    assert res.n_entries == 1
    assert len(res.trades) == 1  # completed round trip despite the block
    assert res.n_open_at_end == 0


def test_direction_specific_masks_gate_only_one_side():
    panel = _pair_panel()
    no_long = simulate_pair(
        panel, long_entry_allowed=np.zeros(len(panel), dtype=bool)
    )
    no_short = simulate_pair(
        panel, short_entry_allowed=np.zeros(len(panel), dtype=bool)
    )
    assert set(no_long.trades["side"]) <= {-1}
    assert set(no_short.trades["side"]) <= {1}


def test_entry_mask_length_is_validated():
    panel = _pair_panel()
    with pytest.raises(ValueError, match="long_entry_allowed"):
        simulate_pair(panel, long_entry_allowed=np.ones(3, dtype=bool))


# --------------------------------------------------------------------------------------
# Short-selling bans
# --------------------------------------------------------------------------------------


def test_short_ban_window_edges_are_inclusive():
    # CNMV blanket ban on Spanish shares, 2020-03-17 to 2020-05-18.
    assert not short_banned("SAN.MC", "2020-03-16")
    assert short_banned("SAN.MC", "2020-03-17")
    assert short_banned("SAN.MC", "2020-05-18")
    assert not short_banned("SAN.MC", "2020-05-19")


def test_blanket_scope_covers_venue_while_named_scope_does_not():
    # 2020 Italian ban is blanket: utilities are caught as well as banks.
    assert short_banned("ENEL.MI", "2020-04-01")
    assert short_banned("ISP.MI", "2020-04-01")
    # 2011 Italian ban named financials only: utilities untouched.
    assert short_banned("ISP.MI", "2011-10-01")
    assert not short_banned("ENEL.MI", "2011-10-01")


def test_amsterdam_and_xetra_were_never_banned():
    """AFM and BaFin declined in 2020, so these pools stay fully tradable."""
    for ticker in ("ASML.AS", "ASM.AS", "BESI.AS", "BMW.DE", "MBG.DE", "VOW3.DE"):
        assert not short_banned(ticker, "2020-04-01")
        assert not short_banned(ticker, "2011-10-01")


def test_no_ban_records_touch_universes_a_to_e():
    """No-damage guarantee: A-E have no records, so masks are all-True on all dates."""
    dates = pd.date_range("2007-01-01", "2024-12-31", freq="B")
    for label in _UNBANNED_UNIVERSES:
        for ticker in pool_tickers(load_s2_pools(label)):
            assert not any(ban.covers(ticker) for ban in SHORT_BANS), ticker
            allow_long, allow_short = pair_entry_masks(ticker, ticker, dates)
            assert allow_long.all() and allow_short.all()


def test_pair_entry_masks_block_only_the_direction_needing_the_banned_leg():
    dates = pd.to_datetime(["2020-04-01"])
    # y banned, x clean: short spread shorts y, so only short is blocked.
    allow_long, allow_short = pair_entry_masks("SAN.MC", "ASML.AS", dates)
    assert allow_long[0] and not allow_short[0]

    # x banned, y clean: long spread shorts x, so only long is blocked.
    allow_long, allow_short = pair_entry_masks("ASML.AS", "SAN.MC", dates)
    assert not allow_long[0] and allow_short[0]

    # Both banned: no entries either way.
    allow_long, allow_short = pair_entry_masks("SAN.MC", "BBVA.MC", dates)
    assert not allow_long[0] and not allow_short[0]

    # Neither banned: unrestricted.
    allow_long, allow_short = pair_entry_masks("ASML.AS", "ASM.AS", dates)
    assert allow_long[0] and allow_short[0]


def test_ban_and_rotation_masks_compose_by_and():
    dates = pd.to_datetime(["2020-04-01", "2020-04-02"])
    rotation = np.array([False, True])
    allow_long, _ = pair_entry_masks("ASML.AS", "ASM.AS", dates)
    combined = rotation & allow_long
    assert list(combined) == [False, True]


# --------------------------------------------------------------------------------------
# Rebalance schedule and slot weighting
# --------------------------------------------------------------------------------------


def test_quarter_end_rebalance_dates_use_real_sessions():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    rebals = quarter_end_rebalance_dates(dates)
    assert len(rebals) == 4
    # Last business day of each quarter, never a weekend or missing session.
    assert [d.month for d in rebals] == [3, 6, 9, 12]
    assert all(d in set(dates) for d in rebals)


def test_schedule_effective_date_is_the_next_session():
    dates = pd.DatetimeIndex(pd.date_range("2020-01-01", "2020-12-31", freq="B"))
    # The final quarter-end has no session after it, so it can never be traded and is
    # dropped (matching build_active_schedule).
    tradable = [r for r in quarter_end_rebalance_dates(dates) if (dates > r).any()]
    assert len(tradable) == 3
    schedule = pd.DataFrame(
        {
            "rebalance_date": tradable,
            "effective_date": [dates[dates > r][0] for r in tradable],
            "pair_id": ["A1|A2"] * len(tradable),
            "pool": ["a"] * len(tradable),
            "pvalue": [0.01] * len(tradable),
            "rank": [1] * len(tradable),
        }
    )
    masks, log = schedule_to_masks(schedule, dates)
    first_effective = schedule["effective_date"].iloc[0]
    mask = masks["A1|A2"]
    # Nothing tradable before the first effective open.
    assert not mask[dates < first_effective].any()
    assert mask[dates >= first_effective].all()
    assert len(log) == len(tradable)


def test_slot_weight_is_invariant_to_active_count():
    """1/6 per slot: a one-pair quarter must not lever up to a six-pair quarter's risk."""
    assert SLOT_WEIGHT == pytest.approx(1.0 / CAP_GLOBAL)
    panel = _pair_panel()
    one = simulate_frozen_book(panel, ["1398.HK|0939.HK"])
    solo = simulate_pair(panel).returns
    pd.testing.assert_series_equal(
        one["returns"],
        (solo * SLOT_WEIGHT).rename("ret"),
        check_names=False,
    )


# --------------------------------------------------------------------------------------
# Trailing-window screen
# --------------------------------------------------------------------------------------


def test_lookback_bars_restricts_the_screen_window():
    n = 400
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    y = pd.Series(x.to_numpy() * 1.5 + rng.normal(0, 0.5, n), index=idx)
    closes = {"AAA": y, "BBB": x}

    full = screen_pair_cointegration(
        closes, [("AAA", "BBB")], is_end=idx[-1]
    )
    trailing = screen_pair_cointegration(
        closes, [("AAA", "BBB")], is_end=idx[-1], lookback_bars=252
    )
    assert int(full["n_is_bars"].iloc[0]) == n
    assert int(trailing["n_is_bars"].iloc[0]) == 252

    # Below the min-bars floor the pair is ineligible, not an EG failure.
    short = screen_pair_cointegration(
        closes, [("AAA", "BBB")], is_end=idx[-1], lookback_bars=100
    )
    assert not bool(short["eligible"].iloc[0])
    assert np.isnan(float(short["pvalue"].iloc[0]))

    with pytest.raises(ValueError, match="lookback_bars"):
        screen_pair_cointegration(
            closes, [("AAA", "BBB")], is_end=idx[-1], lookback_bars=1
        )


def test_build_active_schedule_selects_within_pools_only():
    n = 700
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(11)
    base = np.cumsum(rng.normal(0, 1, n))
    closes = {
        "AAA": pd.Series(100 + base, index=idx),
        "BBB": pd.Series(100 + base + rng.normal(0, 0.4, n), index=idx),
        "CCC": pd.Series(50 + np.cumsum(rng.normal(0, 1, n)), index=idx),
    }
    pools = {"p1": ["AAA", "BBB"], "p2": ["CCC"]}
    schedule = build_active_schedule(closes, pools, idx)
    if not schedule.empty:
        assert set(schedule["pool"]) <= {"p1"}
        assert set(schedule["pair_id"]) <= {"AAA|BBB", "BBB|AAA"}
        # Effective dates always strictly after their rebalance date.
        assert (schedule["effective_date"] > schedule["rebalance_date"]).all()


# --------------------------------------------------------------------------------------
# Cost routing
# --------------------------------------------------------------------------------------


def test_us_alpaca_routing_and_leg_cost():
    for ticker in ("GOOGL", "GOOG", "AMT", "O", "BF.B", "HEI.A", "CWEN.A"):
        assert market_profile_for_ticker(ticker) == "US_ALPACA"
    # Commission-free: 0.1 bps regulatory + 3.2 bps execution.
    assert leg_cost_bps("US_ALPACA", "GOOGL", 100.0) == pytest.approx(3.3)
    # Four leg-fills per pair round trip.
    assert 4 * leg_cost_bps("US_ALPACA", "GOOGL", 100.0) == pytest.approx(13.2)


def test_eur_ibkr_routing_and_leg_cost():
    for ticker in ("SAN.MC", "ISP.MI", "ASML.AS", "BMW.DE", "MC.PA"):
        assert market_profile_for_ticker(ticker) == "F_EUR_IBKR"
    assert leg_cost_bps("F_EUR_IBKR", "SAN.MC", 100.0) == pytest.approx(10.5)


def test_existing_routing_is_unchanged():
    assert market_profile_for_ticker("AUDUSD=X") == "A_FX_OANDA"
    assert market_profile_for_ticker("BTC-USD") == "B_CRYPTO_KRAKEN"
    assert market_profile_for_ticker("0939.HK") == "C_HK_IBKR"
    assert market_profile_for_ticker("8306.T") == "C_JP_IBKR"


def test_unknown_ticker_still_raises():
    with pytest.raises(ValueError, match="unknown market profile"):
        market_profile_for_ticker("1234.XYZ")
