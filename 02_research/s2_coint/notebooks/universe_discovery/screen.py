"""Large-universe cointegration discovery screen (method.md staged gates).

Research-only helper under ``02_research/s2_coint/notebooks/universe_discovery``.
Reuses project EG / book / cost APIs; does not modify the STAR stack.

Pipeline (pre-registered constants — not searched):
  1. within-leaf pairs
  2. shortability (both legs)
  3. liquidity (ADV USD)
  4. corr > CORR_MIN
  5. Engle-Granger + Benjamini–Hochberg FDR
  6. round-trip cost << typical |z|=2 move
  7. persistence (% ADF p < 0.05)
  8. discovery half-life < HL_MAX
  9. select_book under caps
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from data.processing.feature_implementation.cointegration import (
    COINT_PVALUE,
    ou_half_life,
    rolling_adf_pvalue,
    rolling_hedge,
    to_log_price,
)
from data.processing.s2_coint_store import screen_pair_cointegration
from data.processing.s2_universe import iter_pool_pairs, pool_of_pair, pool_tickers
from strategies.s2_coint.book import (
    CAP_GLOBAL,
    CAP_PER_POOL,
    DISCOVERY_LOOKBACK_BARS,
    select_book,
)
from strategies.s2_coint.costs import leg_cost_bps, market_profile_for_ticker

# ---------------------------------------------------------------------------
# Pre-registered discovery constants (not searched)
# ---------------------------------------------------------------------------

OLS_WINDOW = 252
LOOKBACK_BARS = DISCOVERY_LOOKBACK_BARS  # 252
CORR_MIN = 0.80
FDR_Q = 0.10
HL_MAX = 40.0
PERSISTENCE_FLOOR = 0.40
ADF_P_THRESHOLD = COINT_PVALUE  # 0.05
ADF_WINDOW = 126  # persistence rolling ADF (shorter than L so % days is defined)
ADV_WINDOW = 20
MIN_ADV_USD = 5_000_000.0
COST_TO_MOVE_MAX = 0.25  # RT cost / (|z|=2 move) must be < this
COST_PROFILE_DEFAULT = "US_ALPACA"
Z_ENTRY = 2.0

_SCHEDULE_COLS: tuple[str, ...] = (
    "rebalance_date",
    "effective_date",
    "pair_id",
    "pool",
    "pvalue",
    "rank",
)

# Marker for a month-end retest that selected zero pairs (cash / flat book).
EMPTY_BOOK_PAIR_ID = ""

__all__ = [
    "ADF_P_THRESHOLD",
    "ADF_WINDOW",
    "ADV_WINDOW",
    "CORR_MIN",
    "COST_PROFILE_DEFAULT",
    "COST_TO_MOVE_MAX",
    "EMPTY_BOOK_PAIR_ID",
    "FDR_Q",
    "HL_MAX",
    "LOOKBACK_BARS",
    "MIN_ADV_USD",
    "OLS_WINDOW",
    "PERSISTENCE_FLOOR",
    "Z_ENTRY",
    "active_pair_ids",
    "benjamini_hochberg_mask",
    "build_monthly_schedule",
    "funnel_counts",
    "month_end_rebalance_dates",
    "monthly_schedule_to_masks",
    "screen_universe",
    "simulate_monthly_rotating_book",
]


def benjamini_hochberg_mask(
    pvalues: Sequence[float] | pd.Series,
    *,
    q: float = FDR_Q,
) -> np.ndarray:
    """Benjamini–Hochberg FDR: True where the null is rejected at level ``q``."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx], kind="mergesort")]
    m = len(order)
    thresh = q * (np.arange(1, m + 1, dtype=float) / float(m))
    below = p[order] <= thresh
    if not below.any():
        return out
    # Largest k with p_(k) <= q*k/m; reject all i <= k.
    k_max = int(np.max(np.where(below)[0]))
    out[order[: k_max + 1]] = True
    return out


def month_end_rebalance_dates(
    dates: Sequence,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> list[pd.Timestamp]:
    """Last available session of each calendar month on the trading calendar."""
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)))).sort_values().unique()
    idx = pd.DatetimeIndex(idx)
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    if len(idx) == 0:
        return []
    frame = pd.DataFrame({"date": idx})
    frame["month"] = frame["date"].dt.to_period("M")
    return [pd.Timestamp(d) for d in frame.groupby("month")["date"].max().tolist()]


def _next_session(dates: pd.DatetimeIndex, after: pd.Timestamp) -> pd.Timestamp | None:
    later = dates[dates > pd.Timestamp(after)]
    return pd.Timestamp(later[0]) if len(later) else None


def _closes_from_ohlc(
    ohlc_by_ticker: Mapping[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for t, df in ohlc_by_ticker.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        s = df["close"].astype(float).copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        s.name = str(t)
        out[str(t)] = s
    return out


def _slice_to_asof(
    series: pd.Series,
    asof: pd.Timestamp,
    *,
    lookback_bars: int | None,
) -> pd.Series:
    s = series.loc[series.index <= pd.Timestamp(asof)]
    if lookback_bars is not None:
        s = s.iloc[-int(lookback_bars) :]
    return s


def _median_adv_usd(
    ohlc: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    window: int = ADV_WINDOW,
) -> float:
    if ohlc is None or ohlc.empty:
        return float("nan")
    d = ohlc.copy()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    d = d.loc[d.index <= pd.Timestamp(asof)]
    if len(d) < 1:
        return float("nan")
    tail = d.iloc[-int(window) :]
    if "close" not in tail.columns or "volume" not in tail.columns:
        return float("nan")
    dollar = tail["close"].astype(float) * tail["volume"].astype(float)
    med = float(dollar.median())
    return med if np.isfinite(med) else float("nan")


def _pair_corr(
    close_a: pd.Series,
    close_b: pd.Series,
    asof: pd.Timestamp,
    *,
    lookback_bars: int,
) -> float:
    a = _slice_to_asof(close_a, asof, lookback_bars=lookback_bars)
    b = _slice_to_asof(close_b, asof, lookback_bars=lookback_bars)
    aligned = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(aligned) < 3:
        return float("nan")
    ra = aligned.iloc[:, 0].pct_change()
    rb = aligned.iloc[:, 1].pct_change()
    ok = ra.notna() & rb.notna()
    if int(ok.sum()) < 2:
        return float("nan")
    return float(ra.loc[ok].corr(rb.loc[ok]))


def _pair_rt_cost_bps(
    ticker_y: str,
    ticker_x: str,
    *,
    price_y: float,
    price_x: float,
    cost_profile: str | None,
) -> float:
    """Pair round-trip cost in bps of notional (entry+exit on both legs)."""
    py = cost_profile or market_profile_for_ticker(ticker_y)
    px = cost_profile or market_profile_for_ticker(ticker_x)
    # Entry + exit each leg → 2x per leg; sum both legs.
    cy = leg_cost_bps(py, ticker_y, float(price_y))
    cx = leg_cost_bps(px, ticker_x, float(price_x))
    return 2.0 * (float(cy) + float(cx))


def _z2_move_bps(spread: pd.Series) -> float:
    """Typical |z|=2 one-way move in bps of log-spread residual scale."""
    s = spread.astype(float).dropna()
    if len(s) < 3:
        return float("nan")
    sig = float(s.std(ddof=1))
    if not np.isfinite(sig) or sig <= 0:
        return float("nan")
    # Log-spread residual ≈ relative mispricing; 1 unit ≈ 10_000 bps of log-diff.
    return float(Z_ENTRY * sig * 10_000.0)


def _persistence_pct(
    spread: pd.Series,
    *,
    adf_window: int = ADF_WINDOW,
    p_threshold: float = ADF_P_THRESHOLD,
) -> float:
    p = rolling_adf_pvalue(spread, window=adf_window)
    finite = p[np.isfinite(p.to_numpy(dtype=float))]
    if finite.empty:
        return float("nan")
    return float((finite < float(p_threshold)).mean())


def _lookback_spread(
    close_y: pd.Series,
    close_x: pd.Series,
    asof: pd.Timestamp,
    *,
    lookback_bars: int,
    ols_window: int,
) -> pd.Series:
    """PIT rolling OLS spread ending at ``asof``.

    Pulls ``lookback_bars + ols_window - 1`` bars so the trailing ``lookback_bars``
    residuals are defined (rolling OLS needs a full window before the first finite spread).
    """
    need = int(lookback_bars) + int(ols_window) - 1
    y = _slice_to_asof(close_y, asof, lookback_bars=need)
    x = _slice_to_asof(close_x, asof, lookback_bars=need)
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(aligned) < max(ols_window, 3):
        return pd.Series(dtype=float)
    y_log = to_log_price(aligned.iloc[:, 0])
    x_log = to_log_price(aligned.iloc[:, 1])
    hedge = rolling_hedge(y_log, x_log, window=ols_window)
    spread = hedge["spread"].dropna()
    if len(spread) > int(lookback_bars):
        spread = spread.iloc[-int(lookback_bars) :]
    return spread


def funnel_counts(screen: pd.DataFrame) -> pd.Series:
    """Count rows by ``exclude_reason`` (empty string = passed all gates / book-eligible)."""
    if screen is None or screen.empty or "exclude_reason" not in screen.columns:
        return pd.Series(dtype=int)
    reasons = screen["exclude_reason"].fillna("").astype(str)
    # Book selection tags survivors that fail caps separately when annotate is used;
    # here empty means still in play after HL gate.
    vc = reasons.value_counts(dropna=False)
    vc.name = "n"
    return vc


def screen_universe(
    ohlc_by_ticker: Mapping[str, pd.DataFrame],
    pools,
    *,
    asof: pd.Timestamp | str,
    lookback_bars: int = LOOKBACK_BARS,
    ols_window: int = OLS_WINDOW,
    corr_min: float = CORR_MIN,
    fdr_q: float = FDR_Q,
    hl_max: float = HL_MAX,
    persistence_floor: float = PERSISTENCE_FLOOR,
    min_adv_usd: float = MIN_ADV_USD,
    adv_window: int = ADV_WINDOW,
    cost_to_move_max: float = COST_TO_MOVE_MAX,
    cost_profile: str | None = COST_PROFILE_DEFAULT,
    shortable_by_ticker: Mapping[str, bool] | None = None,
    skip_shortability: bool = False,
    per_pool_cap: int = CAP_PER_POOL,
    global_cap: int = CAP_GLOBAL,
    pvalue_threshold: float = COINT_PVALUE,
) -> pd.DataFrame:
    """Staged discovery screen at calendar ``asof`` (uses only data with date <= asof).

    Returns one row per unordered candidate (EG may re-orient ``pair_id``). Columns include
    gate diagnostics and ``exclude_reason`` / ``passed`` / ``auto_selected``.
    """
    asof_ts = pd.Timestamp(asof)
    closes = _closes_from_ohlc(ohlc_by_ticker)
    candidates = iter_pool_pairs(pools)
    pools_by_pair = pool_of_pair(pools)
    tickers = pool_tickers(pools)

    # --- ticker-level gates ---
    short_ok: dict[str, bool] = {}
    for t in tickers:
        if skip_shortability or shortable_by_ticker is None:
            short_ok[t] = True
        else:
            short_ok[t] = bool(shortable_by_ticker.get(t, False))

    adv_ok: dict[str, bool] = {}
    adv_usd: dict[str, float] = {}
    for t in tickers:
        med = _median_adv_usd(
            ohlc_by_ticker.get(t, pd.DataFrame()),
            asof_ts,
            window=adv_window,
        )
        adv_usd[t] = med
        adv_ok[t] = bool(np.isfinite(med) and med >= float(min_adv_usd))

    # Pre-filter candidates that fail ticker gates (before expensive EG).
    pre_pairs: list[tuple[str, str]] = []
    pre_rows: list[dict] = []
    for a, b in candidates:
        pid_ab = f"{a}|{b}"
        base = {
            "pair_id": pid_ab,
            "ticker_a": a,
            "ticker_b": b,
            "pool": pools_by_pair.get(pid_ab, ""),
            "corr": float("nan"),
            "pvalue": float("nan"),
            "tstat": float("nan"),
            "discovery_half_life": float("nan"),
            "n_is_bars": 0,
            "eligible": False,
            "fdr_pass": False,
            "adv_usd_a": adv_usd.get(a, float("nan")),
            "adv_usd_b": adv_usd.get(b, float("nan")),
            "rt_cost_bps": float("nan"),
            "z2_move_bps": float("nan"),
            "cost_to_move": float("nan"),
            "pct_adf_lt_005": float("nan"),
            "exclude_reason": "",
            "passed": False,
            "auto_selected": False,
        }
        if not short_ok.get(a, False) or not short_ok.get(b, False):
            base["exclude_reason"] = "shortability"
            pre_rows.append(base)
            continue
        if not adv_ok.get(a, False) or not adv_ok.get(b, False):
            base["exclude_reason"] = "liquidity"
            pre_rows.append(base)
            continue
        if a not in closes or b not in closes:
            base["exclude_reason"] = "missing_price"
            pre_rows.append(base)
            continue
        corr = _pair_corr(closes[a], closes[b], asof_ts, lookback_bars=lookback_bars)
        base["corr"] = corr
        if (not np.isfinite(corr)) or corr < float(corr_min):
            base["exclude_reason"] = "corr"
            pre_rows.append(base)
            continue
        pre_pairs.append((a, b))
        pre_rows.append(base)

    # EG on corr survivors only.
    eg = (
        screen_pair_cointegration(
            closes,
            pre_pairs,
            is_end=asof_ts,
            ols_window=ols_window,
            pvalue_threshold=pvalue_threshold,
            lookback_bars=lookback_bars,
        )
        if pre_pairs
        else pd.DataFrame()
    )
    eg_by_slot: dict[str, pd.Series] = {}
    if not eg.empty:
        for row in eg.itertuples(index=False):
            parts = sorted(str(row.pair_id).split("|"))
            eg_by_slot["|".join(parts)] = row

    # Merge EG into pre_rows that reached corr pass.
    merged: list[dict] = []
    for base in pre_rows:
        if base["exclude_reason"]:
            merged.append(base)
            continue
        a, b = str(base["ticker_a"]), str(base["ticker_b"])
        slot = "|".join(sorted((a, b)))
        eg_row = eg_by_slot.get(slot)
        if eg_row is None:
            base["exclude_reason"] = "eg_missing"
            merged.append(base)
            continue
        base["pair_id"] = str(eg_row.pair_id)
        base["ticker_y"] = str(eg_row.ticker_y)
        base["ticker_x"] = str(eg_row.ticker_x)
        base["pvalue"] = float(eg_row.pvalue) if pd.notna(eg_row.pvalue) else float("nan")
        base["tstat"] = float(eg_row.tstat) if pd.notna(eg_row.tstat) else float("nan")
        base["discovery_half_life"] = (
            float(eg_row.discovery_half_life)
            if pd.notna(eg_row.discovery_half_life)
            else float("nan")
        )
        base["n_is_bars"] = int(eg_row.n_is_bars)
        base["eligible"] = bool(eg_row.eligible)
        if not base["eligible"]:
            base["exclude_reason"] = "ineligible"
        elif not (np.isfinite(base["pvalue"]) and base["pvalue"] < float(pvalue_threshold)):
            base["exclude_reason"] = "not_passer"
        merged.append(base)

    df = pd.DataFrame(merged)
    if df.empty:
        return df

    # Ensure orientation columns exist for early rejects.
    if "ticker_y" not in df.columns:
        df["ticker_y"] = df.get("ticker_a", "")
    if "ticker_x" not in df.columns:
        df["ticker_x"] = df.get("ticker_b", "")

    # FDR among eligible raw EG passers only.
    passer_mask = (
        df["exclude_reason"].astype(str).eq("")
        & df["eligible"].astype(bool)
        & df["pvalue"].notna()
    )
    fdr_flags = np.zeros(len(df), dtype=bool)
    if passer_mask.any():
        idxs = np.where(passer_mask.to_numpy())[0]
        bh = benjamini_hochberg_mask(df.loc[passer_mask, "pvalue"].to_numpy(), q=fdr_q)
        for i, flag in zip(idxs, bh):
            fdr_flags[i] = bool(flag)
    df["fdr_pass"] = fdr_flags
    for i in np.where(passer_mask.to_numpy() & ~fdr_flags)[0]:
        df.at[i, "exclude_reason"] = "fdr"

    # Cost / persistence / HL on FDR passers.
    for i, row in df.iterrows():
        if str(row["exclude_reason"]):
            continue
        ty, tx = str(row["ticker_y"]), str(row["ticker_x"])
        if ty not in closes or tx not in closes:
            df.at[i, "exclude_reason"] = "missing_price"
            continue
        spread = _lookback_spread(
            closes[ty],
            closes[tx],
            asof_ts,
            lookback_bars=lookback_bars,
            ols_window=ols_window,
        )
        # Gate HL on the trailing residual path (EG scalar HL is often NaN when
        # lookback_bars == ols_window because only one rolling residual exists).
        gate_hl = ou_half_life(spread.dropna())
        if np.isfinite(gate_hl):
            df.at[i, "discovery_half_life"] = float(gate_hl)

        z2 = _z2_move_bps(spread.dropna())
        py = float(closes[ty].loc[closes[ty].index <= asof_ts].iloc[-1])
        px = float(closes[tx].loc[closes[tx].index <= asof_ts].iloc[-1])
        rt = _pair_rt_cost_bps(
            ty, tx, price_y=py, price_x=px, cost_profile=cost_profile
        )
        ratio = (
            float(rt) / float(z2)
            if np.isfinite(rt) and np.isfinite(z2) and z2 > 0
            else float("nan")
        )
        df.at[i, "rt_cost_bps"] = rt
        df.at[i, "z2_move_bps"] = z2
        df.at[i, "cost_to_move"] = ratio
        if (not np.isfinite(ratio)) or ratio >= float(cost_to_move_max):
            df.at[i, "exclude_reason"] = "cost"
            continue

        pers = _persistence_pct(spread.dropna())
        df.at[i, "pct_adf_lt_005"] = pers
        if (not np.isfinite(pers)) or pers < float(persistence_floor):
            df.at[i, "exclude_reason"] = "persistence"
            continue

        hl = float(df.at[i, "discovery_half_life"])
        if (not np.isfinite(hl)) or hl <= 0.0 or hl >= float(hl_max):
            df.at[i, "exclude_reason"] = "half_life"
            continue

        df.at[i, "passed"] = True

    # Book selection among passers.
    survivors = df.loc[df["passed"].astype(bool)].copy()
    if survivors.empty:
        df["auto_selected"] = False
        return df.reset_index(drop=True)

    # select_book expects screen-like columns.
    book = select_book(
        survivors,
        pool_of_pair=pools_by_pair,
        per_pool_cap=per_pool_cap,
        global_cap=global_cap,
        pvalue_threshold=pvalue_threshold,
    )
    selected = set(book["pair_id"].astype(str)) if not book.empty else set()
    df["auto_selected"] = df["pair_id"].astype(str).isin(selected)
    # Tag passers that lost on caps.
    cap_mask = df["passed"].astype(bool) & ~df["auto_selected"].astype(bool)
    df.loc[cap_mask, "exclude_reason"] = df.loc[cap_mask, "exclude_reason"].where(
        df.loc[cap_mask, "exclude_reason"].astype(str).ne(""),
        "book_cap",
    )
    return df.reset_index(drop=True)


def build_monthly_schedule(
    ohlc_by_ticker: Mapping[str, pd.DataFrame],
    pools,
    dates: Sequence,
    *,
    lookback_bars: int = LOOKBACK_BARS,
    ols_window: int = OLS_WINDOW,
    per_pool_cap: int = CAP_PER_POOL,
    global_cap: int = CAP_GLOBAL,
    rebalance_dates: Sequence | None = None,
    shortable_by_ticker: Mapping[str, bool] | None = None,
    skip_shortability: bool = False,
    cost_profile: str | None = COST_PROFILE_DEFAULT,
    **screen_kwargs,
) -> pd.DataFrame:
    """Monthly rotating book: full gated screen at each month-end, effective next session.

    Every month-end with a following session is recorded. Months with no EG/gate
    survivors get a single marker row (``pair_id == EMPTY_BOOK_PAIR_ID``) so the
    active book goes flat rather than silently keeping the prior month's pairs.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)))).sort_values().unique()
    idx = pd.DatetimeIndex(idx)
    rebals = (
        [pd.Timestamp(d) for d in rebalance_dates]
        if rebalance_dates is not None
        else month_end_rebalance_dates(idx)
    )
    rows: list[dict] = []
    for t in rebals:
        effective = _next_session(idx, t)
        if effective is None:
            continue
        screen = screen_universe(
            ohlc_by_ticker,
            pools,
            asof=t,
            lookback_bars=lookback_bars,
            ols_window=ols_window,
            per_pool_cap=per_pool_cap,
            global_cap=global_cap,
            shortable_by_ticker=shortable_by_ticker,
            skip_shortability=skip_shortability,
            cost_profile=cost_profile,
            **screen_kwargs,
        )
        chosen = screen.loc[screen["auto_selected"].astype(bool)].copy()
        if chosen.empty:
            rows.append(
                {
                    "rebalance_date": t,
                    "effective_date": effective,
                    "pair_id": EMPTY_BOOK_PAIR_ID,
                    "pool": "",
                    "pvalue": float("nan"),
                    "rank": 0,
                }
            )
            continue
        chosen = chosen.sort_values(["pvalue", "pair_id"], kind="mergesort").reset_index(
            drop=True
        )
        chosen["rank"] = range(1, len(chosen) + 1)
        for row in chosen.itertuples(index=False):
            rows.append(
                {
                    "rebalance_date": t,
                    "effective_date": effective,
                    "pair_id": str(row.pair_id),
                    "pool": str(getattr(row, "pool", "")),
                    "pvalue": float(row.pvalue),
                    "rank": int(row.rank),
                }
            )
    return pd.DataFrame(rows, columns=list(_SCHEDULE_COLS))


def active_pair_ids(schedule: pd.DataFrame) -> list[str]:
    """Real pair ids on a monthly schedule (excludes empty-book markers)."""
    if schedule is None or schedule.empty or "pair_id" not in schedule.columns:
        return []
    ids = []
    seen: set[str] = set()
    for pid in schedule["pair_id"].astype(str):
        p = str(pid).strip()
        if not p or p == EMPTY_BOOK_PAIR_ID:
            continue
        if p in seen:
            continue
        seen.add(p)
        ids.append(p)
    return ids


def monthly_schedule_to_masks(
    schedule: pd.DataFrame,
    panel_dates: Sequence,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Like ``schedule_to_masks``, but empty-book months demote all pairs (flat)."""
    from strategies.s2_coint.book import BookState, apply_rebalance

    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(panel_dates)))).sort_values().unique()
    idx = pd.DatetimeIndex(idx)
    n = len(idx)

    all_pairs = active_pair_ids(schedule)
    masks: dict[str, np.ndarray] = {p: np.zeros(n, dtype=bool) for p in all_pairs}
    log_cols = [
        "rebalance_date",
        "effective_date",
        "n_active",
        "promoted",
        "demoted",
        "flipped",
    ]
    if schedule is None or schedule.empty or n == 0:
        return masks, pd.DataFrame(columns=log_cols)

    state = BookState()
    log: list[dict] = []
    groups = list(schedule.groupby("effective_date", sort=True))

    for gi, (effective, chunk) in enumerate(groups):
        selection = [
            str(p).strip()
            for p in chunk["pair_id"].tolist()
            if str(p).strip() and str(p).strip() != EMPTY_BOOK_PAIR_ID
        ]
        open_pairs = set(state.active.values())
        moves = apply_rebalance(state, selection, open_pairs=open_pairs)

        start = int(idx.searchsorted(pd.Timestamp(effective), side="left"))
        stop = (
            int(idx.searchsorted(pd.Timestamp(groups[gi + 1][0]), side="left"))
            if gi + 1 < len(groups)
            else n
        )
        for pid in state.active.values():
            if pid not in masks:
                masks[pid] = np.zeros(n, dtype=bool)
            masks[pid][start:stop] = True

        log.append(
            {
                "rebalance_date": pd.Timestamp(chunk["rebalance_date"].iloc[0]),
                "effective_date": pd.Timestamp(effective),
                "n_active": len(state.active),
                "promoted": ",".join(moves["promoted"]),
                "demoted": ",".join(moves["demoted"]),
                "flipped": ",".join(moves["flipped"]),
            }
        )

    return masks, pd.DataFrame(log)


def simulate_monthly_rotating_book(
    panel: pd.DataFrame,
    schedule: pd.DataFrame,
    cfg=None,
    *,
    slot_weight: float | None = None,
    apply_short_bans: bool = False,
) -> dict:
    """Monthly rotate sim; empty-book months are flat (no new entries).

    Same return schema as ``simulate_rotating_book``. When the schedule has only
    empty markers (never any pairs), returns calendar-aligned zero returns if
    ``panel`` has dates, else an empty series.
    """
    from backtest.s2_coint.rotation import SLOT_WEIGHT
    from strategies.s2_coint.config import S2SimConfig
    from strategies.s2_coint.engine import simulate_pair
    from strategies.s2_coint.metrics import metrics_from_returns
    from strategies.s2_coint.short_bans import pair_entry_masks

    cfg = cfg or S2SimConfig()
    w = SLOT_WEIGHT if slot_weight is None else float(slot_weight)
    pair_ids = active_pair_ids(schedule)

    if not pair_ids:
        if panel is not None and not panel.empty and "date" in panel.columns:
            dates = pd.DatetimeIndex(pd.to_datetime(panel["date"])).sort_values().unique()
            ret = pd.Series(0.0, index=dates, dtype=float, name="ret")
        else:
            ret = pd.Series(dtype=float, name="ret")
        return {
            "arm": "rotate_monthly",
            "returns": ret,
            "pair_returns": {},
            "trades": pd.DataFrame(),
            "metrics": metrics_from_returns(ret),
            "rebalance_log": monthly_schedule_to_masks(schedule, ret.index)[1]
            if len(ret)
            else pd.DataFrame(),
        }

    # Reuse rotating sim path but with empty-aware masks.
    import numpy as np

    dates = pd.DatetimeIndex(
        pd.to_datetime(panel["date"]).sort_values().unique()
    )
    masks, log = monthly_schedule_to_masks(schedule, dates)

    per_pair: dict[str, pd.Series] = {}
    trades: list[pd.DataFrame] = []
    for pid, mask in masks.items():
        g = panel.loc[panel["pair_id"].astype(str) == str(pid)].copy()
        if g.empty:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        pos = dates.searchsorted(pd.DatetimeIndex(pd.to_datetime(g["date"])))
        pair_mask = mask[np.clip(pos, 0, len(mask) - 1)]
        if not pair_mask.any():
            continue
        if apply_short_bans:
            ban_long, ban_short = pair_entry_masks(g)
            allow_long = pair_mask & ban_long
            allow_short = pair_mask & ban_short
        else:
            allow_long = pair_mask
            allow_short = pair_mask
        res = simulate_pair(
            g,
            cfg,
            long_entry_allowed=allow_long,
            short_entry_allowed=allow_short,
        )
        per_pair[str(pid)] = res.returns
        if not res.trades.empty:
            trades.append(res.trades)

    if not per_pair:
        ret = pd.Series(0.0, index=dates, dtype=float, name="ret")
    else:
        wide = pd.concat(
            [s.rename(pid) for pid, s in per_pair.items() if not s.empty], axis=1
        )
        ret = (wide.reindex(dates).fillna(0.0).sum(axis=1) * float(w)).rename("ret")

    return {
        "arm": "rotate_monthly",
        "returns": ret,
        "pair_returns": per_pair,
        "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        "metrics": metrics_from_returns(ret),
        "rebalance_log": log,
    }