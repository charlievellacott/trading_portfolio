"""S3 pair/book simulator: close-t signal → fill open t+1 (open-to-open PnL)."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategies.s3_fx_trend.config import S3SimConfig
from strategies.s3_fx_trend.costs import (
    daily_swap_return,
    leg_cost_bps,
    pair_supported,
    resolve_cost_profile,
)
from strategies.s3_fx_trend.overlays import (
    apply_overlay_gate,
    apply_overlay_tilt,
    carry_signal_weekly,
    value_signal_monthly,
)
from strategies.s3_fx_trend.sizing import (
    apply_conviction,
    apply_rebalance_band,
    apply_weak_signal_flat,
    book_vol_target_leverage,
    enforce_currency_cap,
    pair_inverse_vol_weights,
)

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252.0

_TRADE_COLS: tuple[str, ...] = (
    "pair",
    "side",
    "entry_date",
    "exit_date",
    "hold_bars",
    "entry_cost_bps",
    "exit_cost_bps",
    "exit_reason",
)

_BLEND_LOOKBACK = {
    "single_1m": 21,
    "single_3m": 63,
    "single_12m": 252,
}


@dataclass
class PairSimResult:
    """Net/gross returns, weights, and trades for one FX pair."""

    pair: str
    returns_net: pd.Series
    returns_gross: pd.Series
    trades: pd.DataFrame
    weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass
class BookSimResult:
    """Book-level net/gross returns plus per-pair results."""

    returns_net: pd.Series
    returns_gross: pd.Series
    trades: pd.DataFrame
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    pair_results: dict[str, PairSimResult] = field(default_factory=dict)
    leverage: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _normalize_pair(pair: str) -> str:
    return str(pair).strip().upper().replace("_", "").replace("/", "").replace("=X", "")


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_TRADE_COLS))


def _panel_pair_id(d: pd.DataFrame) -> str:
    for col in ("pair", "pair_id", "ticker"):
        if col in d.columns and len(d):
            return _normalize_pair(str(d[col].iloc[0]))
    return ""


def _close_series(d: pd.DataFrame) -> pd.Series:
    if "close" in d.columns:
        s = pd.to_numeric(d["close"], errors="coerce")
    elif "Close" in d.columns:
        s = pd.to_numeric(d["Close"], errors="coerce")
    else:
        raise ValueError("panel_pair requires a close column")
    s.index = pd.to_datetime(d["date"]) if "date" in d.columns else d.index
    return s.astype(float)


def _open_series(d: pd.DataFrame) -> pd.Series:
    if "open" in d.columns:
        s = pd.to_numeric(d["open"], errors="coerce")
    elif "Open" in d.columns:
        s = pd.to_numeric(d["Open"], errors="coerce")
    else:
        raise ValueError("panel_pair requires an open column")
    s.index = pd.to_datetime(d["date"]) if "date" in d.columns else d.index
    return s.astype(float)


def _tsmom_raw(close: pd.Series, lookbacks: tuple[int, ...], blend: str) -> pd.Series:
    if blend == "equal_blend":
        parts = []
        for lb in lookbacks:
            parts.append(close / close.shift(int(lb)) - 1.0)
        stacked = pd.concat(parts, axis=1)
        # Equal blend of horizon returns, then keep continuous magnitude.
        return stacked.mean(axis=1)
    lb = _BLEND_LOOKBACK.get(blend)
    if lb is None:
        lb = int(lookbacks[-1]) if lookbacks else 252
    return close / close.shift(int(lb)) - 1.0


def _donchian_permission(d: pd.DataFrame, close: pd.Series, n: int) -> pd.Series:
    """+1 breakout up, -1 breakout down, else 0. Uses precomputed cols when present."""
    up_col = f"donch_break_up_{n}"
    dn_col = f"donch_break_dn_{n}"
    hi_col = f"donch_high_{n}"
    lo_col = f"donch_low_{n}"
    if up_col in d.columns and dn_col in d.columns:
        up = pd.to_numeric(d[up_col], errors="coerce").astype(float)
        dn = pd.to_numeric(d[dn_col], errors="coerce").astype(float)
        up.index = close.index
        dn.index = close.index
        sig = pd.Series(0.0, index=close.index, dtype=float)
        sig = sig.where(~(up > 0), 1.0)
        sig = sig.where(~(dn > 0), -1.0)
        # Prefer up if both (rare).
        both = (up > 0) & (dn > 0)
        sig = sig.where(~both, 1.0)
        return sig
    if hi_col in d.columns and lo_col in d.columns:
        hi = pd.to_numeric(d[hi_col], errors="coerce").astype(float)
        lo = pd.to_numeric(d[lo_col], errors="coerce").astype(float)
        hi.index = close.index
        lo.index = close.index
    else:
        # Prior N bars only (exclude current close from channel).
        hi = close.shift(1).rolling(int(n), min_periods=int(n)).max()
        lo = close.shift(1).rolling(int(n), min_periods=int(n)).min()
    sig = pd.Series(0.0, index=close.index, dtype=float)
    sig = sig.mask(close > hi, 1.0)
    sig = sig.mask(close < lo, -1.0)
    return sig


def compute_signal(panel_pair: pd.DataFrame, cfg: S3SimConfig) -> pd.Series:
    """Signal known at close ``t`` in roughly ``[-1, +1]`` (pre-conviction)."""
    if panel_pair is None or panel_pair.empty:
        return pd.Series(dtype=float, name="signal")
    d = panel_pair.sort_values("date").reset_index(drop=True) if "date" in panel_pair.columns else panel_pair.copy()
    close = _close_series(d)
    family = cfg.signal_family
    tsmom = _tsmom_raw(close, cfg.tsmom_lookbacks, cfg.tsmom_blend)
    # Prefer precomputed tsmom columns when blend is a single horizon.
    blend_lb = _BLEND_LOOKBACK.get(cfg.tsmom_blend)
    if blend_lb is not None:
        col = f"tsmom_{blend_lb}"
        if col in d.columns:
            tsmom = pd.to_numeric(d[col], errors="coerce").astype(float)
            tsmom.index = close.index

    if family == "tsmom":
        raw = tsmom
    elif family == "donchian":
        raw = _donchian_permission(d, close, int(cfg.donchian_n))
    else:
        # both: TSMOM direction × Donchian permission (flat when no break / disagree)
        don = _donchian_permission(d, close, int(cfg.donchian_n))
        direction = np.sign(tsmom)
        raw = direction.where(
            (don != 0) & (np.sign(don) == direction),
            0.0,
        )

    raw = apply_weak_signal_flat(raw, cfg.weak_signal_flat_z)
    out = apply_conviction(raw, cfg.conviction_scaling)
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=close.index, dtype=float)
    out.name = "signal"
    return out


def _rate_diff_on_date(
    pair: str,
    rates_df: pd.DataFrame | None,
    asof: pd.Timestamp,
) -> float:
    if rates_df is None or rates_df.empty:
        return 0.0
    p = _normalize_pair(pair)
    base, quote = p[:3], p[3:]
    cols = {str(c).upper(): c for c in rates_df.columns}
    if base not in cols or quote not in cols:
        return 0.0
    idx = pd.to_datetime(rates_df.index)
    sub = rates_df.copy()
    sub.index = idx
    sub = sub.sort_index()
    sub = sub.loc[sub.index <= pd.Timestamp(asof)]
    if sub.empty:
        return 0.0
    row = sub.iloc[-1]
    b = float(pd.to_numeric(row[cols[base]], errors="coerce"))
    q = float(pd.to_numeric(row[cols[quote]], errors="coerce"))
    if not (np.isfinite(b) and np.isfinite(q)):
        return 0.0
    return b - q


def simulate_pair(
    panel_pair: pd.DataFrame,
    cfg: S3SimConfig,
    *,
    rates_df: pd.DataFrame | None = None,
) -> PairSimResult:
    """Core pair sim: signal close ``t``, fill open ``t+1``, PnL open-to-open.

    Weight decided at close ``t`` is held from open[t+1] → open[t+2]; return is
    attributed to the fill date ``t+1``. Turnover costs on ``|Δw|``; optional swap.
    """
    empty = PairSimResult(
        pair="",
        returns_net=pd.Series(dtype=float),
        returns_gross=pd.Series(dtype=float),
        trades=_empty_trades(),
        weights=pd.Series(dtype=float),
    )
    if panel_pair is None or panel_pair.empty:
        return empty

    d = panel_pair.sort_values("date").reset_index(drop=True)
    pair = _panel_pair_id(d)
    if not pair or not pair_supported(pair):
        if pair:
            logger.warning("skipping unsupported pair %s", pair)
        return empty

    resolve_cost_profile(pair, cfg.cost_profile)
    signal = compute_signal(d, cfg)
    dates = pd.to_datetime(d["date"]).to_list()
    opens = _open_series(d).to_numpy(dtype=float)
    closes = _close_series(d).to_numpy(dtype=float)
    sig = signal.reindex(pd.to_datetime(d["date"])).to_numpy(dtype=float)

    # Optional pair VT: scale |signal| by inverse-vol leverage from close-to-close.
    c2c = pd.Series(closes, index=pd.to_datetime(d["date"])).pct_change()
    lev = pd.Series(1.0, index=c2c.index, dtype=float)
    if cfg.pair_vt:
        vt = pair_inverse_vol_weights(
            c2c.to_frame(name=pair),
            target_ann_vol=cfg.vt_target_ann_vol,
        )
        if pair in vt.columns:
            lev = vt[pair].reindex(c2c.index).fillna(1.0)

    target_w = pd.Series(sig, index=pd.to_datetime(d["date"]), dtype=float) * lev
    target_w = target_w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Carry overlay on the pair weight path (value overlay is book-level).
    if cfg.carry_mode != "off" and rates_df is not None and not rates_df.empty:
        carry = carry_signal_weekly(pair, rates_df)
        carry = carry.reindex(target_w.index).ffill()
        if cfg.carry_mode == "gate":
            target_w = apply_overlay_gate(target_w, carry)
        elif cfg.carry_mode == "tilt":
            target_w = apply_overlay_tilt(
                target_w,
                carry,
                cfg.carry_agree_scale,
                cfg.carry_disagree_scale,
            )

    n = len(d)
    w_held = 0.0
    net_by_date: dict[pd.Timestamp, float] = defaultdict(float)
    gross_by_date: dict[pd.Timestamp, float] = defaultdict(float)
    weight_rows: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []
    open_trade: dict | None = None

    for i in range(n - 2):
        # Signal at close t=i → weight applies from open[i+1] to open[i+2].
        tgt = float(target_w.iloc[i]) if np.isfinite(target_w.iloc[i]) else 0.0
        fill_date = pd.Timestamp(dates[i + 1])
        exit_date = pd.Timestamp(dates[i + 2])
        px_fill = float(opens[i + 1]) if np.isfinite(opens[i + 1]) else float(closes[i])

        applied = apply_rebalance_band(w_held, tgt, cfg.rebalance_band)
        dw = applied - w_held
        cost = abs(dw) * leg_cost_bps(pair, px_fill) / 10_000.0

        # Trade ledger on stance changes through zero or flips.
        if w_held == 0.0 and applied != 0.0:
            open_trade = {
                "side": int(np.sign(applied)),
                "entry_date": fill_date,
                "entry_idx": i + 1,
                "entry_cost_bps": abs(dw) * leg_cost_bps(pair, px_fill),
            }
        elif w_held != 0.0 and applied == 0.0 and open_trade is not None:
            trades.append(
                {
                    "pair": pair,
                    "side": int(open_trade["side"]),
                    "entry_date": open_trade["entry_date"],
                    "exit_date": fill_date,
                    "hold_bars": int((i + 1) - int(open_trade["entry_idx"])),
                    "entry_cost_bps": float(open_trade["entry_cost_bps"]),
                    "exit_cost_bps": abs(dw) * leg_cost_bps(pair, px_fill),
                    "exit_reason": "flat",
                }
            )
            open_trade = None
        elif w_held != 0.0 and applied != 0.0 and np.sign(w_held) != np.sign(applied):
            if open_trade is not None:
                trades.append(
                    {
                        "pair": pair,
                        "side": int(open_trade["side"]),
                        "entry_date": open_trade["entry_date"],
                        "exit_date": fill_date,
                        "hold_bars": int((i + 1) - int(open_trade["entry_idx"])),
                        "entry_cost_bps": float(open_trade["entry_cost_bps"]),
                        "exit_cost_bps": abs(w_held) * leg_cost_bps(pair, px_fill),
                        "exit_reason": "flip",
                    }
                )
            open_trade = {
                "side": int(np.sign(applied)),
                "entry_date": fill_date,
                "entry_idx": i + 1,
                "entry_cost_bps": abs(applied) * leg_cost_bps(pair, px_fill),
            }

        w_held = applied
        weight_rows[fill_date] = w_held

        o0, o1 = opens[i + 1], opens[i + 2]
        if np.isfinite(o0) and np.isfinite(o1) and o0 != 0:
            asset_ret = o1 / o0 - 1.0
        else:
            asset_ret = 0.0
        gross = w_held * asset_ret
        swap = 0.0
        if cfg.include_swap and w_held != 0.0:
            rd = _rate_diff_on_date(pair, rates_df, fill_date)
            swap = daily_swap_return(
                pair,
                w_held,
                rd,
                financing_spread_bps_annual=cfg.financing_spread_bps_annual,
                wednesday_triple=True,
                asof=fill_date,
            )
        net_by_date[fill_date] += gross + swap - cost
        gross_by_date[fill_date] += gross

    net = pd.Series(net_by_date, dtype=float).sort_index()
    gross_s = pd.Series(gross_by_date, dtype=float).sort_index()
    net.name = pair
    gross_s.name = pair
    weights = pd.Series(weight_rows, dtype=float).sort_index()
    weights.name = pair
    return PairSimResult(
        pair=pair,
        returns_net=net,
        returns_gross=gross_s,
        trades=pd.DataFrame(trades, columns=list(_TRADE_COLS)),
        weights=weights,
    )


def _spike_scale(
    returns: pd.Series,
    *,
    threshold_sigmas: float,
    window: int = 60,
) -> pd.Series:
    """1.0 normally; 0.5 when |ret| exceeds trailing vol × threshold."""
    r = pd.to_numeric(returns, errors="coerce").astype(float)
    vol = r.rolling(window, min_periods=max(10, window // 3)).std(ddof=1)
    # PIT: compare today's return to yesterday's vol estimate.
    vol_pit = vol.shift(1)
    scale = pd.Series(1.0, index=r.index, dtype=float)
    thr = float(threshold_sigmas)
    if thr <= 0:
        return scale
    spike = r.abs() > (thr * vol_pit)
    scale = scale.where(~spike.fillna(False), 0.5)
    return scale


def simulate_book(
    panel: pd.DataFrame,
    cfg: S3SimConfig,
    *,
    rates_df: pd.DataFrame | None = None,
    reer_df: pd.DataFrame | None = None,
) -> BookSimResult:
    """Equal-weight (then optional book VT / caps / overlays) across pairs."""
    empty = BookSimResult(
        returns_net=pd.Series(dtype=float, name="ret"),
        returns_gross=pd.Series(dtype=float, name="ret_gross"),
        trades=_empty_trades(),
    )
    if panel is None or panel.empty:
        return empty

    pair_col = "pair" if "pair" in panel.columns else (
        "pair_id" if "pair_id" in panel.columns else None
    )
    if pair_col is None:
        # Single-pair panel.
        res = simulate_pair(panel, cfg, rates_df=rates_df)
        return BookSimResult(
            returns_net=res.returns_net.rename("ret"),
            returns_gross=res.returns_gross.rename("ret_gross"),
            trades=res.trades,
            weights=res.weights.to_frame(res.pair) if not res.weights.empty else pd.DataFrame(),
            pair_results={res.pair: res} if res.pair else {},
        )

    pairs = cfg.pairs or tuple(
        sorted({_normalize_pair(p) for p in panel[pair_col].unique()})
    )
    pair_results: dict[str, PairSimResult] = {}
    net_parts: list[pd.Series] = []
    gross_parts: list[pd.Series] = []
    weight_parts: list[pd.Series] = []
    trade_frames: list[pd.DataFrame] = []

    for pair in pairs:
        p = _normalize_pair(pair)
        if not pair_supported(p):
            logger.warning("simulate_book: skipping unsupported pair %s", p)
            continue
        g = panel.loc[panel[pair_col].map(_normalize_pair) == p].copy()
        if g.empty:
            continue
        # Value overlay: apply on signal via a one-off cfg path inside pair by
        # pre-multiplying is awkward; do post-weight tilt below.
        res = simulate_pair(g, cfg, rates_df=rates_df)
        if not res.pair:
            continue
        pair_results[res.pair] = res
        if not res.returns_net.empty:
            net_parts.append(res.returns_net.rename(res.pair))
            gross_parts.append(res.returns_gross.rename(res.pair))
        if not res.weights.empty:
            weight_parts.append(res.weights.rename(res.pair))
        if not res.trades.empty:
            trade_frames.append(res.trades)

    if not net_parts:
        return empty

    net_wide = pd.concat(net_parts, axis=1).fillna(0.0)
    gross_wide = pd.concat(gross_parts, axis=1).fillna(0.0)
    # Equal-weight across active pairs.
    book_net = net_wide.mean(axis=1).rename("ret")
    book_gross = gross_wide.mean(axis=1).rename("ret_gross")

    weights = (
        pd.concat(weight_parts, axis=1).fillna(0.0) if weight_parts else pd.DataFrame()
    )

    # Book-level value overlay tilt/gate on weights (then leave returns as-is —
    # research notebooks re-simulate when overlay STARs change). Applied here as
    # a post-scale on pair returns for gate/tilt approximation.
    if cfg.value_mode != "off" and reer_df is not None and not reer_df.empty and not weights.empty:
        scaled_net = []
        scaled_gross = []
        for col in net_wide.columns:
            val = value_signal_monthly(col, reer_df, refresh=cfg.value_refresh)
            val = val.reindex(net_wide.index).ffill()
            wcol = weights[col] if col in weights.columns else pd.Series(0.0, index=net_wide.index)
            if cfg.value_mode == "gate":
                gated = apply_overlay_gate(wcol, val)
                mult = (gated.abs() > 0).astype(float)
                # When gated flat, zero that pair's contribution.
                scaled_net.append((net_wide[col] * mult).rename(col))
                scaled_gross.append((gross_wide[col] * mult).rename(col))
            else:
                tilted = apply_overlay_tilt(
                    wcol,
                    val,
                    cfg.value_agree_scale,
                    cfg.value_disagree_scale,
                )
                # Scale returns by tilt / original weight ratio.
                ratio = tilted.abs() / wcol.abs().replace(0.0, np.nan)
                ratio = ratio.fillna(1.0).clip(0.0, 5.0)
                scaled_net.append((net_wide[col] * ratio).rename(col))
                scaled_gross.append((gross_wide[col] * ratio).rename(col))
        net_wide = pd.concat(scaled_net, axis=1).fillna(0.0)
        gross_wide = pd.concat(scaled_gross, axis=1).fillna(0.0)
        book_net = net_wide.mean(axis=1).rename("ret")
        book_gross = gross_wide.mean(axis=1).rename("ret_gross")

    # Currency cap: shrink weights then rescale pair returns approximately.
    if cfg.corr_conflict_mode in {"currency_cap", "both"} and not weights.empty:
        capped_rows = []
        for dt, row in weights.iterrows():
            capped_rows.append(enforce_currency_cap(row, cfg.currency_cap_gross_pct))
        weights = pd.DataFrame(capped_rows)

    # Spike delever on book and/or pairs.
    lev_series = pd.Series(1.0, index=book_net.index, dtype=float)
    if cfg.spike_mode in {"book_delever", "both"}:
        lev_series = lev_series * _spike_scale(
            book_net, threshold_sigmas=cfg.spike_threshold_sigmas
        )
    if cfg.spike_mode in {"pair_delever", "both"}:
        for col in list(net_wide.columns):
            sc = _spike_scale(net_wide[col], threshold_sigmas=cfg.spike_threshold_sigmas)
            net_wide[col] = net_wide[col] * sc
            gross_wide[col] = gross_wide[col] * sc
        book_net = net_wide.mean(axis=1).rename("ret")
        book_gross = gross_wide.mean(axis=1).rename("ret_gross")

    if cfg.book_vt:
        # Causal scalar leverage path: expand history day by day.
        lev_vals: list[float] = []
        prev: float | None = None
        hist: list[float] = []
        for dt, r in book_net.items():
            L = book_vol_target_leverage(
                hist,
                cfg.vt_target_ann_vol,
                prev_leverage=prev,
            )
            lev_vals.append(L)
            prev = L
            if np.isfinite(r):
                hist.append(float(r))
        lev_series = lev_series * pd.Series(lev_vals, index=book_net.index, dtype=float)
        book_net = (book_net * lev_series).rename("ret")
        book_gross = (book_gross * lev_series).rename("ret_gross")

    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else _empty_trades()
    )
    return BookSimResult(
        returns_net=book_net,
        returns_gross=book_gross,
        trades=trades,
        weights=weights,
        pair_results=pair_results,
        leverage=lev_series.rename("leverage"),
    )


def _pair_sign_for_currency(pair: str, ccy: str) -> int:
    """+1 if stronger ``ccy`` ⇒ long pair; -1 if stronger ⇒ short; 0 if unrelated."""
    p = _normalize_pair(pair)
    c = str(ccy).strip().upper()
    if len(p) != 6:
        return 0
    base, quote = p[:3], p[3:]
    if c == base:
        return 1
    if c == quote:
        return -1
    return 0


def simulate_event_trades(
    event_panel: pd.DataFrame,
    price_bars: pd.DataFrame,
    cfg: S3SimConfig,
    *,
    bar: str | None = None,
) -> BookSimResult:
    """Event sleeve: fill at first bar open strictly after print τ; hold N bars.

    ``event_panel`` needs ``date`` (τ), ``z``, ``impact``, ``currency`` (or ``pair``).
    ``price_bars`` long panel with ``date``, ``pair``, ``open`` (and ideally ``close``).
    """
    empty = BookSimResult(
        returns_net=pd.Series(dtype=float, name="ret"),
        returns_gross=pd.Series(dtype=float, name="ret_gross"),
        trades=_empty_trades(),
    )
    if (
        event_panel is None
        or event_panel.empty
        or price_bars is None
        or price_bars.empty
    ):
        return empty

    bar_s = str(bar or cfg.event_bar)
    hold = int(cfg.event_hold_bars)
    z_gate = float(cfg.event_z_gate)
    impact_min = int(cfg.event_impact_min)

    ev = event_panel.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    if "z" not in ev.columns:
        return empty
    ev["z"] = pd.to_numeric(ev["z"], errors="coerce")
    if "impact" in ev.columns:
        ev["impact"] = pd.to_numeric(ev["impact"], errors="coerce").fillna(0)
    else:
        ev["impact"] = impact_min
    ev = ev.loc[(ev["z"].abs() >= z_gate) & (ev["impact"] >= impact_min)].copy()
    if ev.empty:
        return empty

    prices = price_bars.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    pair_col = "pair" if "pair" in prices.columns else "pair_id"
    prices[pair_col] = prices[pair_col].map(_normalize_pair)
    prices = prices.sort_values(["date", pair_col])

    pairs = [_normalize_pair(p) for p in (cfg.pairs or tuple(prices[pair_col].unique()))]
    pairs = [p for p in pairs if pair_supported(p)]

    net_by_date: dict[pd.Timestamp, float] = defaultdict(float)
    gross_by_date: dict[pd.Timestamp, float] = defaultdict(float)
    trades: list[dict] = []

    for _, row in ev.iterrows():
        tau = pd.Timestamp(row["date"])
        z = float(row["z"])
        if not np.isfinite(z):
            continue
        ccy = str(row["currency"]).upper() if "currency" in ev.columns else ""
        target_pairs = []
        if "pair" in ev.columns and pd.notna(row.get("pair")):
            target_pairs = [_normalize_pair(str(row["pair"]))]
        else:
            for p in pairs:
                if _pair_sign_for_currency(p, ccy) != 0:
                    target_pairs.append(p)
        for pair in target_pairs:
            if not pair_supported(pair):
                continue
            psign = _pair_sign_for_currency(pair, ccy) if ccy else 1
            if psign == 0:
                continue
            size = float(np.sign(z) * psign)
            if cfg.surprise_sizing:
                size *= float(np.clip(abs(z) / max(z_gate, 1e-9), 0.5, 2.0))

            bars = prices.loc[prices[pair_col] == pair].copy()
            if bars.empty:
                continue
            # First bar whose open timestamp is strictly after τ.
            future = bars.loc[bars["date"] > tau]
            if future.empty:
                continue
            fill_i = future.index[0]
            fill_pos = bars.index.get_loc(fill_i)
            if isinstance(fill_pos, slice):
                continue
            fill_pos = int(fill_pos)
            exit_pos = fill_pos + hold
            if exit_pos >= len(bars):
                continue
            fill_row = bars.iloc[fill_pos]
            exit_row = bars.iloc[exit_pos]
            o0 = float(pd.to_numeric(fill_row["open"], errors="coerce"))
            o1 = float(pd.to_numeric(exit_row["open"], errors="coerce"))
            if not (np.isfinite(o0) and np.isfinite(o1) and o0 != 0):
                continue
            fill_date = pd.Timestamp(fill_row["date"])
            exit_date = pd.Timestamp(exit_row["date"])
            # Assert fill after τ (contract).
            if fill_date <= tau:
                continue
            asset_ret = o1 / o0 - 1.0
            cost = abs(size) * leg_cost_bps(pair, o0) / 10_000.0
            # Exit cost on flatten.
            cost += abs(size) * leg_cost_bps(pair, o1) / 10_000.0
            gross = size * asset_ret
            net = gross - cost
            net_by_date[fill_date] += net
            gross_by_date[fill_date] += gross
            trades.append(
                {
                    "pair": pair,
                    "side": int(np.sign(size)),
                    "entry_date": fill_date,
                    "exit_date": exit_date,
                    "hold_bars": hold,
                    "entry_cost_bps": abs(size) * leg_cost_bps(pair, o0),
                    "exit_cost_bps": abs(size) * leg_cost_bps(pair, o1),
                    "exit_reason": f"event_hold_{bar_s}",
                }
            )

    net = pd.Series(net_by_date, dtype=float).sort_index().rename("ret")
    gross = pd.Series(gross_by_date, dtype=float).sort_index().rename("ret_gross")
    return BookSimResult(
        returns_net=net,
        returns_gross=gross,
        trades=pd.DataFrame(trades, columns=list(_TRADE_COLS)),
    )
