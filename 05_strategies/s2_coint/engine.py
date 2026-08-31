"""S2 pair/book simulator: close-t signal, fill open t+1 (research, not Strategy)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from risk.s1_equities.vol_targeting import VolTargetConfig, leverage_from_history
from strategies.s2_coint.baseline import (
    PERIODS_PER_YEAR,
    PairSimResult,
    _TRADE_COLS,
)
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.costs import (
    daily_borrow_return,
    leg_cost_bps,
    market_profile_for_pair,
    resolve_cost_profile,
)
from strategies.s2_coint.overlap import (
    corr_blocks_candidate,
    pair_tickers,
    score_confidence,
    shares_leg,
)
from strategies.s2_coint.metrics import book_returns_to_calendar, panel_session_dates
from strategies.s2_coint.sizing import atr_size_multiplier, pair_scale_from_score

_EMPTY_TRADES = pd.DataFrame(columns=list(_TRADE_COLS))


def _mean_abs_at_entry(
    g: pd.DataFrame,
    i: int,
    fallback: float,
    *,
    column: str = "mean_abs_score",
) -> float:
    """PIT row ``mean_abs_score`` when present; else scalar ``fallback``."""
    if column in g.columns:
        v = float(g[column].iloc[i])
        if np.isfinite(v) and v > 0:
            return v
    fb = float(fallback)
    return fb if np.isfinite(fb) and fb > 0 else 1.0


def _entry_mask(
    mask: Sequence[bool] | np.ndarray | None,
    n: int,
    name: str,
) -> np.ndarray:
    """Normalize an optional entry mask to a length-n bool array (None => all allowed)."""
    if mask is None:
        return np.ones(n, dtype=bool)
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have length {n}, got {arr.shape}")
    return arr


@dataclass
class BookSimResult:
    """Equal-weight (or scale-weighted) book returns plus per-pair trades."""

    returns: pd.Series
    pair_results: dict[str, PairSimResult] = field(default_factory=dict)
    leverage: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _score_series(d: pd.DataFrame, cfg: S2SimConfig) -> np.ndarray:
    col = cfg.score_column
    if cfg.entry_mode == "v3_hmm_innov" and "z_innov" in d.columns:
        col = "z_innov"
    elif cfg.entry_mode == "v1_ewm_asym" and "z_ewm" in d.columns:
        col = "z_ewm"
    elif cfg.entry_mode == "v2_ou" and "ou_score" in d.columns:
        col = "ou_score"
    if col not in d.columns:
        col = "z"
    return d[col].to_numpy(dtype=float)


def _k_in(cfg: S2SimConfig, sigma_t: float) -> float:
    if cfg.vol_mode == "kt":
        bar = cfg.sigma_bar if cfg.sigma_bar is not None and cfg.sigma_bar > 0 else 1.0
        if np.isfinite(sigma_t) and sigma_t > 0:
            return float(cfg.k0) * float(sigma_t) / float(bar)
        return float("nan")
    if cfg.entry_mode in {"v1_roll_asym", "v1_ewm_asym", "v2_ou"}:
        return float(cfg.k_in)
    return float(cfg.entry_z)


def _k_out(cfg: S2SimConfig) -> float:
    if cfg.entry_mode in {"v1_roll_asym", "v1_ewm_asym", "v2_ou"}:
        return float(cfg.k_out)
    return float(cfg.exit_z)


def _health_blocks_entry(row_adf: float, row_vj: float, row_hl: float, cfg: S2SimConfig) -> bool:
    if cfg.break_mode == "block_05_flat_10":
        if np.isfinite(row_adf) and row_adf >= 0.05:
            return True
    elif cfg.break_mode == "flat_05":
        if np.isfinite(row_adf) and row_adf >= 0.05:
            return True
    if cfg.hl_gate_min is not None or cfg.hl_gate_max is not None:
        if not np.isfinite(row_hl):
            return True
        if cfg.hl_gate_min is not None and row_hl < cfg.hl_gate_min:
            return True
        if cfg.hl_gate_max is not None and row_hl > cfg.hl_gate_max:
            return True
    _ = row_vj
    return False


def _health_flattens(row_adf: float, row_vj: float, cfg: S2SimConfig) -> bool:
    if cfg.break_mode == "block_05_flat_10":
        if np.isfinite(row_adf) and row_adf >= 0.10:
            return True
        if np.isfinite(row_vj) and row_vj >= 2.0:
            return True
    elif cfg.break_mode == "flat_05":
        if np.isfinite(row_adf) and row_adf >= 0.05:
            return True
    return False


def _trend_blocks(side: int, rsi: float, adx: float, cfg: S2SimConfig) -> bool:
    mode = cfg.trend_mode
    if mode == "off":
        return False
    if mode in {"adx_veto", "both"}:
        if np.isfinite(adx) and adx > 25.0:
            return True
    if mode in {"rsi_confirm", "both"}:
        if side > 0 and (not np.isfinite(rsi) or rsi >= 30.0):
            return True
        if side < 0 and (not np.isfinite(rsi) or rsi <= 70.0):
            return True
    return False


def _hmm_blocks(d: pd.DataFrame, i: int, cfg: S2SimConfig) -> bool:
    if cfg.entry_mode != "v3_hmm_innov":
        return False
    if "p_mr" not in d.columns:
        return True
    p = float(d["p_mr"].to_numpy(dtype=float)[i])
    return (not np.isfinite(p)) or p < float(cfg.hmm_mr_threshold)


def _breaker_reentry_ok(row_adf: float, row_hl: float, cfg: S2SimConfig) -> bool:
    if not (np.isfinite(row_adf) and row_adf < 0.05):
        return False
    lo = cfg.hl_gate_min if cfg.hl_gate_min is not None else 5.0
    hi = cfg.hl_gate_max if cfg.hl_gate_max is not None else 60.0
    return np.isfinite(row_hl) and lo <= row_hl <= hi


def _classify_exit_reason(
    *,
    mean_revert: bool,
    coint_break: bool,
    hl_timeout: bool = False,
    max_loss: bool = False,
    atr_stop: bool = False,
) -> str:
    """Pick one blotter label; health/break beats mean-revert when both fire."""
    if coint_break:
        return "coint_break"
    if max_loss:
        return "max_loss"
    if atr_stop:
        return "atr_stop"
    if hl_timeout:
        return "hl_timeout"
    if mean_revert:
        return "mean_revert"
    return "other"


def simulate_pair(
    df: pd.DataFrame,
    cfg: S2SimConfig | None = None,
    *,
    mean_abs_score: float = 1.0,
    n_pairs: int = 1,
    leverage: float = 1.0,
    long_entry_allowed: Sequence[bool] | np.ndarray | None = None,
    short_entry_allowed: Sequence[bool] | np.ndarray | None = None,
) -> PairSimResult:
    """One pair: decide at close t, fill both legs at open t+1.

    ``long_entry_allowed`` / ``short_entry_allowed`` are optional per-row boolean masks
    (aligned to ``df`` after sorting by date) gating **new entries only**; exits are never
    gated, so an open position always runs to its normal z-exit. Both default to None
    (= allowed), leaving behaviour identical to a call without them.

    Two callers use this:

    - H-004 rotation passes the same mask to both directions, so a demoted pair (or the old
      side of an orientation flip) stops opening but still exits.
    - ``short_bans.pair_entry_masks`` passes direction-specific masks, because a ban on one
      leg only blocks the spread direction that needs to short it.
    """
    cfg = cfg or S2SimConfig()
    empty = PairSimResult(
        pair_id="",
        returns=pd.Series(dtype=float),
        trades=_EMPTY_TRADES.copy(),
        n_entries=0,
        n_open_at_end=0,
        open_entry_cost_bps=0.0,
    )
    if df.empty:
        return empty

    d = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(d["date"]).to_list()
    score = _score_series(d, cfg)
    beta = d[cfg.beta_column].to_numpy(dtype=float)
    oy = d["open_y"].to_numpy(dtype=float)
    ox = d["open_x"].to_numpy(dtype=float)
    hy = d["high_y"].to_numpy(dtype=float) if "high_y" in d.columns else oy
    ly = d["low_y"].to_numpy(dtype=float) if "low_y" in d.columns else oy
    hx = d["high_x"].to_numpy(dtype=float) if "high_x" in d.columns else ox
    lx = d["low_x"].to_numpy(dtype=float) if "low_x" in d.columns else ox
    hl = d["half_life"].to_numpy(dtype=float) if "half_life" in d.columns else np.full(len(d), np.nan)
    adf = (
        d["adf_pvalue"].to_numpy(dtype=float)
        if "adf_pvalue" in d.columns
        else np.full(len(d), np.nan)
    )
    vj = (
        d["variance_jump"].to_numpy(dtype=float)
        if "variance_jump" in d.columns
        else np.full(len(d), np.nan)
    )
    rsi = (
        d["rsi_spread"].to_numpy(dtype=float)
        if "rsi_spread" in d.columns
        else np.full(len(d), np.nan)
    )
    adx = (
        d["adx_spread"].to_numpy(dtype=float)
        if "adx_spread" in d.columns
        else np.full(len(d), np.nan)
    )
    atr = (
        d["atr_spread"].to_numpy(dtype=float)
        if "atr_spread" in d.columns
        else np.full(len(d), np.nan)
    )
    sig = (
        d["spread"].astype(float).rolling(cfg.sigma_window, min_periods=cfg.sigma_window).std(ddof=1)
        .to_numpy(dtype=float)
        if "spread" in d.columns
        else np.full(len(d), np.nan)
    )
    sh = d["spread_high"].to_numpy(dtype=float) if "spread_high" in d.columns else np.full(len(d), np.nan)
    sl = d["spread_low"].to_numpy(dtype=float) if "spread_low" in d.columns else np.full(len(d), np.nan)
    so = d["spread_open"].to_numpy(dtype=float) if "spread_open" in d.columns else np.full(len(d), np.nan)

    allow_long = _entry_mask(long_entry_allowed, len(d), "long_entry_allowed")
    allow_short = _entry_mask(short_entry_allowed, len(d), "short_entry_allowed")

    ty = str(d["ticker_y"].iloc[0])
    tx = str(d["ticker_x"].iloc[0])
    pair_id = str(d["pair_id"].iloc[0])
    profile = resolve_cost_profile(pair_id, ty, tx, cost_profile=cfg.cost_profile)

    pos = 0
    pnl_by_date: dict[pd.Timestamp, float] = defaultdict(float)
    trades: list[dict] = []
    n_entries = 0
    open_entry: dict | None = None
    breaker_block = False
    k_out = _k_out(cfg)

    for i in range(len(d) - 2):
        z_t = score[i]
        beta_fill = beta[i + 1]
        fill_date = pd.Timestamp(dates[i + 1])
        kin = _k_in(cfg, sig[i])

        do_exit = False
        mean_revert = False
        coint_break = False
        hl_timeout = False
        max_loss = False
        atr_stop = False
        if pos != 0 and np.isfinite(z_t) and (float(pos) * z_t >= float(k_out)):
            do_exit = True
            mean_revert = True
        if pos != 0 and _health_flattens(adf[i], vj[i], cfg):
            do_exit = True
            coint_break = True
        if (
            cfg.exit_mode == "hl3_atr_breaker"
            and open_entry is not None
            and np.isfinite(open_entry.get("hl_at_entry", np.nan))
        ):
            hold = (i + 1) - int(open_entry["entry_idx"])
            if hold >= float(cfg.n_half_lives) * float(open_entry["hl_at_entry"]):
                do_exit = True
                hl_timeout = True
            if float(open_entry.get("cum_pnl", 0.0)) <= float(cfg.pair_max_loss):
                do_exit = True
                max_loss = True
                breaker_block = True
            # Path-check H/L on the fill bar (i+1) after entry open.
            if np.isfinite(open_entry.get("stop_spread", np.nan)) and np.isfinite(sl[i + 1]):
                side = int(open_entry["side"])
                if side > 0 and sl[i + 1] <= float(open_entry["stop_spread"]):
                    do_exit = True
                    atr_stop = True
                if side < 0 and sh[i + 1] >= float(open_entry["stop_spread"]):
                    do_exit = True
                    atr_stop = True

        event_cost = 0.0
        if do_exit and pos != 0:
            by = leg_cost_bps(profile, ty, oy[i + 1])
            bx = leg_cost_bps(profile, tx, ox[i + 1])
            exit_cost_bps = float(by + bx)
            event_cost += exit_cost_bps / 10_000.0
            if open_entry is not None:
                trades.append(
                    {
                        "pair_id": pair_id,
                        "side": int(open_entry["side"]),
                        "entry_date": open_entry["entry_date"],
                        "exit_date": fill_date,
                        "hold_bars": int((i + 1) - int(open_entry["entry_idx"])),
                        "entry_cost_bps": float(open_entry["entry_cost_bps"]),
                        "exit_cost_bps": exit_cost_bps,
                        "exit_reason": _classify_exit_reason(
                            mean_revert=mean_revert,
                            coint_break=coint_break,
                            hl_timeout=hl_timeout,
                            max_loss=max_loss,
                            atr_stop=atr_stop,
                        ),
                    }
                )
            open_entry = None
            pos = 0

        want_long = np.isfinite(z_t) and np.isfinite(kin) and z_t <= -kin and allow_long[i]
        want_short = np.isfinite(z_t) and np.isfinite(kin) and z_t >= kin and allow_short[i]
        side = 1 if want_long else (-1 if want_short else 0)
        do_entry = (
            pos == 0
            and side != 0
            and np.isfinite(beta_fill)
            and not _health_blocks_entry(adf[i], vj[i], hl[i], cfg)
            and not _trend_blocks(side, rsi[i], adx[i], cfg)
            and not _hmm_blocks(d, i, cfg)
        )
        if breaker_block:
            if _breaker_reentry_ok(adf[i], hl[i], cfg):
                breaker_block = False
            else:
                do_entry = False

        scale = pair_scale_from_score(
            z_t,
            adf[i],
            size_mode=cfg.size_mode,
            mean_abs_score=_mean_abs_at_entry(d, i, mean_abs_score),
        )
        size_mult = 1.0
        if cfg.exit_mode == "hl3_atr_breaker" and do_entry:
            size_mult = atr_size_multiplier(
                atr=float(atr[i]) if np.isfinite(atr[i]) else float("nan"),
                beta=float(beta_fill),
                n_pairs=n_pairs,
                pair_scale=scale,
                leverage=leverage,
                risk_frac=cfg.atr_risk_frac,
            )

        if do_entry:
            pos = side
            by = leg_cost_bps(profile, ty, oy[i + 1])
            bx = leg_cost_bps(profile, tx, ox[i + 1])
            entry_cost_bps = float(by + bx)
            event_cost += entry_cost_bps / 10_000.0
            n_entries += 1
            stop_spread = float("nan")
            if np.isfinite(so[i + 1]) and np.isfinite(atr[i]) and atr[i] > 0:
                stop_spread = float(so[i + 1] - pos * atr[i])
            open_entry = {
                "side": pos,
                "entry_idx": i + 1,
                "entry_date": fill_date,
                "entry_cost_bps": entry_cost_bps,
                "hl_at_entry": float(hl[i]) if np.isfinite(hl[i]) else float("nan"),
                "cum_pnl": 0.0,
                "scale": float(scale) * float(size_mult) * float(leverage),
                "stop_spread": stop_spread,
            }

        bar_ret = 0.0
        if pos != 0 and np.isfinite(beta_fill):
            ry = oy[i + 2] / oy[i + 1] - 1.0
            rx = ox[i + 2] / ox[i + 1] - 1.0
            y_w = float(pos)
            x_w = (
                -float(pos) * float(beta_fill) if cfg.use_hedge_ratio_sizing else -float(pos)
            )
            gross = abs(y_w) + abs(x_w)
            if gross > 0:
                y_w /= gross
                x_w /= gross
            bar_ret = y_w * ry + x_w * rx
            if open_entry is not None:
                bar_ret *= float(open_entry.get("scale", 1.0))
                open_entry["cum_pnl"] = float(open_entry.get("cum_pnl", 0.0)) + bar_ret
            bar_ret += daily_borrow_return(
                profile,
                y_weight=y_w,
                x_weight=x_w,
                scale=float(open_entry.get("scale", 1.0)) if open_entry else 1.0,
            )

        pnl_by_date[fill_date] += bar_ret - event_cost

    out = pd.Series(pnl_by_date, dtype=float).sort_index()
    out.name = pair_id
    trade_df = pd.DataFrame(trades, columns=list(_TRADE_COLS))
    open_cost = float(open_entry["entry_cost_bps"]) if open_entry is not None else 0.0
    return PairSimResult(
        pair_id=pair_id,
        returns=out,
        trades=trade_df,
        n_entries=n_entries,
        n_open_at_end=int(open_entry is not None),
        open_entry_cost_bps=open_cost,
    )


def _densify_book_returns(
    raw: pd.Series,
    panel: pd.DataFrame,
    *,
    leverage: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    """Align sparse joint-sim returns to the panel session calendar."""
    cal = panel_session_dates(panel)
    out = book_returns_to_calendar(raw, cal)
    lev_out = None
    if leverage is not None and not leverage.empty:
        lev_out = leverage.astype(float).reindex(out.index).ffill().fillna(1.0)
    return out, lev_out


def simulate_book(
    panel: pd.DataFrame,
    cfg: S2SimConfig | None = None,
    *,
    mean_abs_score: float = 1.0,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> BookSimResult:
    """Book-level sim with optional overlap / corr gates and S1-style VT."""
    cfg = cfg or S2SimConfig()
    if panel.empty:
        return BookSimResult(returns=pd.Series(dtype=float, name="ret"))

    pairs = list(panel.groupby("pair_id", sort=False))
    n_pairs = max(len(pairs), 1)
    lev_hist: list[float] = []
    lev_index: list[pd.Timestamp] = []
    leverage = 1.0
    prev_lev = 1.0
    vt_cfg = VolTargetConfig(
        enabled=cfg.vol_mode == "s1_vt",
        target_ann_vol=cfg.vt_target_ann_vol,
        periods_per_year=float(periods_per_year),
        min_periods=max(13, int(cfg.sigma_window // 4)),
    )

    if cfg.overlap_mode == "allow" and cfg.corr_k is None:
        parts: list[pd.Series] = []
        pair_results: dict[str, PairSimResult] = {}
        for pid, g in pairs:
            res = simulate_pair(
                g,
                cfg,
                mean_abs_score=mean_abs_score,
                n_pairs=n_pairs,
                leverage=1.0,
            )
            pair_results[pid] = res
            if not res.returns.empty:
                parts.append(res.returns.rename(pid))
        if not parts:
            return BookSimResult(returns=pd.Series(dtype=float, name="ret"))
        wide = pd.concat(parts, axis=1).fillna(0.0)
        raw = wide.mean(axis=1).rename("ret")
        if cfg.vol_mode == "s1_vt":
            scaled = []
            lev_vals = []
            for t, r in raw.items():
                leverage = leverage_from_history(
                    pd.Series(lev_hist, index=pd.DatetimeIndex(lev_index)),
                    vt_cfg,
                    prev_leverage=prev_lev,
                )
                scaled.append(float(r) * float(leverage))
                lev_vals.append(leverage)
                lev_hist.append(float(r))
                lev_index.append(pd.Timestamp(t))
                prev_lev = leverage
            out = pd.Series(scaled, index=raw.index, name="ret")
            out, lev_ser = _densify_book_returns(
                out,
                panel,
                leverage=pd.Series(lev_vals, index=raw.index, dtype=float),
            )
            return BookSimResult(
                returns=out,
                pair_results=pair_results,
                leverage=lev_ser,
            )
        out, _ = _densify_book_returns(raw, panel)
        return BookSimResult(returns=out, pair_results=pair_results)

    return _simulate_book_joint(
        panel,
        cfg,
        mean_abs_score=mean_abs_score,
        n_pairs=n_pairs,
        periods_per_year=periods_per_year,
    )


def _simulate_book_joint(
    panel: pd.DataFrame,
    cfg: S2SimConfig,
    *,
    mean_abs_score: float,
    n_pairs: int,
    periods_per_year: float,
) -> BookSimResult:
    """Calendar loop for never-allow overlap and Kalman corr gate."""
    by_pair = {
        str(pid): g.sort_values("date").reset_index(drop=True)
        for pid, g in panel.groupby("pair_id", sort=False)
    }
    # Precompute independent "would trade" via per-pair sim is wrong for overlap.
    # Joint: iterate unique signal dates; apply exits then gated entries.
    pnl: dict[str, dict[pd.Timestamp, float]] = {pid: defaultdict(float) for pid in by_pair}
    trades: dict[str, list] = {pid: [] for pid in by_pair}
    n_entries = {pid: 0 for pid in by_pair}
    rho_cache: dict[tuple[str, str], pd.Series] = {}

    if cfg.corr_k is not None and "spread" in panel.columns:
        from data.processing.feature_implementation.kalman import kalman_correlation

        pids = list(by_pair)
        for i, a in enumerate(pids):
            sa = by_pair[a].set_index("date")["spread"].astype(float).diff()
            for b in pids[i + 1 :]:
                sb = by_pair[b].set_index("date")["spread"].astype(float).diff()
                joined = pd.concat([sa.rename("a"), sb.rename("b")], axis=1).dropna()
                if joined.empty:
                    continue
                rho_cache[(a, b)] = kalman_correlation(joined["a"], joined["b"])

    signal_dates = sorted({pd.Timestamp(t) for g in by_pair.values() for t in g["date"]})
    open_pos: dict[str, dict] = {}
    breaker_block: dict[str, bool] = {pid: False for pid in by_pair}

    def _rho_at(a: str, b: str, ts: pd.Timestamp) -> float | None:
        key = tuple(sorted((a, b)))
        series = rho_cache.get((key[0], key[1]))
        if series is None or ts not in series.index:
            return None
        val = float(series.loc[ts])
        return val if np.isfinite(val) else None

    for sig_date in signal_dates:
        exit_ids: list[tuple[str, str]] = []
        entry_cands: list[tuple[str, int, float]] = []
        for pid, g in by_pair.items():
            rows = g.index[pd.to_datetime(g["date"]) == sig_date]
            if len(rows) == 0:
                continue
            i = int(rows[0])
            if i >= len(g) - 2:
                continue
            score = _score_series(g, cfg)[i]
            adf = float(g["adf_pvalue"].iloc[i]) if "adf_pvalue" in g.columns else float("nan")
            vj = float(g["variance_jump"].iloc[i]) if "variance_jump" in g.columns else float("nan")
            hl = float(g["half_life"].iloc[i]) if "half_life" in g.columns else float("nan")
            rsi = float(g["rsi_spread"].iloc[i]) if "rsi_spread" in g.columns else float("nan")
            adx = float(g["adx_spread"].iloc[i]) if "adx_spread" in g.columns else float("nan")
            sl = float(g["spread_low"].iloc[i + 1]) if "spread_low" in g.columns else float("nan")
            sh = float(g["spread_high"].iloc[i + 1]) if "spread_high" in g.columns else float("nan")
            sig = float("nan")
            if "spread" in g.columns:
                s = g["spread"].astype(float)
                if i + 1 >= cfg.sigma_window:
                    sig = float(s.iloc[i - cfg.sigma_window + 1 : i + 1].std(ddof=1))
            kin = _k_in(cfg, sig)
            kout = _k_out(cfg)
            st = open_pos.get(pid)
            if st is not None:
                pos = int(st["side"])
                mean_revert = np.isfinite(score) and (pos * score >= kout)
                coint_break = _health_flattens(adf, vj, cfg)
                hl_timeout = False
                max_loss = False
                atr_stop = False
                flatten = mean_revert or coint_break
                if cfg.exit_mode == "hl3_atr_breaker":
                    hold = (i + 1) - int(st["entry_idx"])
                    if np.isfinite(st.get("hl_at_entry", np.nan)) and hold >= cfg.n_half_lives * st["hl_at_entry"]:
                        flatten = True
                        hl_timeout = True
                    if float(st.get("cum_pnl", 0.0)) <= cfg.pair_max_loss:
                        flatten = True
                        max_loss = True
                        breaker_block[pid] = True
                    if np.isfinite(st.get("stop_spread", np.nan)) and np.isfinite(sl):
                        if pos > 0 and sl <= float(st["stop_spread"]):
                            flatten = True
                            atr_stop = True
                        if pos < 0 and np.isfinite(sh) and sh >= float(st["stop_spread"]):
                            flatten = True
                            atr_stop = True
                if flatten:
                    exit_ids.append(
                        (
                            pid,
                            _classify_exit_reason(
                                mean_revert=bool(mean_revert),
                                coint_break=bool(coint_break),
                                hl_timeout=hl_timeout,
                                max_loss=max_loss,
                                atr_stop=atr_stop,
                            ),
                        )
                    )
            else:
                want_long = np.isfinite(score) and np.isfinite(kin) and score <= -kin
                want_short = np.isfinite(score) and np.isfinite(kin) and score >= kin
                side = 1 if want_long else (-1 if want_short else 0)
                if breaker_block[pid]:
                    if _breaker_reentry_ok(adf, hl, cfg):
                        breaker_block[pid] = False
                    else:
                        side = 0
                if (
                    side != 0
                    and not _health_blocks_entry(adf, vj, hl, cfg)
                    and not _trend_blocks(side, rsi, adx, cfg)
                    and not _hmm_blocks(g, i, cfg)
                ):
                    sc = score_confidence(score, adf)
                    entry_cands.append((pid, side, sc))

        for pid, exit_reason in exit_ids:
            g = by_pair[pid]
            i = int(g.index[pd.to_datetime(g["date"]) == sig_date][0])
            fill_date = pd.Timestamp(g["date"].iloc[i + 1])
            st = open_pos.pop(pid)
            ty, tx = str(g["ticker_y"].iloc[0]), str(g["ticker_x"].iloc[0])
            profile = resolve_cost_profile(pid, ty, tx, cost_profile=cfg.cost_profile)
            by = leg_cost_bps(profile, ty, float(g["open_y"].iloc[i + 1]))
            bx = leg_cost_bps(profile, tx, float(g["open_x"].iloc[i + 1]))
            pnl[pid][fill_date] -= (by + bx) / 10_000.0
            trades[pid].append(
                {
                    "pair_id": pid,
                    "side": int(st["side"]),
                    "entry_date": st["entry_date"],
                    "exit_date": fill_date,
                    "hold_bars": int((i + 1) - int(st["entry_idx"])),
                    "entry_cost_bps": float(st["entry_cost_bps"]),
                    "exit_cost_bps": float(by + bx),
                    "exit_reason": exit_reason,
                }
            )

        if cfg.overlap_mode == "never_allow" or cfg.corr_k is not None:
            entry_cands.sort(key=lambda t: (t[2], t[0]), reverse=True)

        rho_now: dict[tuple[str, str], float] = {}
        if cfg.corr_k is not None:
            ids = list(open_pos.keys()) + [c[0] for c in entry_cands]
            for a_i, a in enumerate(ids):
                for b in ids[a_i + 1 :]:
                    val = _rho_at(a, b, sig_date)
                    if val is not None:
                        rho_now[tuple(sorted((a, b)))] = val

        for pid, side, sc in entry_cands:
            g = by_pair[pid]
            ty, tx = str(g["ticker_y"].iloc[0]), str(g["ticker_x"].iloc[0])
            legs = pair_tickers(pid, ty, tx)
            blocked = False
            if cfg.overlap_mode == "never_allow":
                for oid, ost in open_pos.items():
                    og = by_pair[oid]
                    olegs = pair_tickers(
                        oid, str(og["ticker_y"].iloc[0]), str(og["ticker_x"].iloc[0])
                    )
                    if shares_leg(legs, olegs):
                        blocked = True
                        break
                if not blocked:
                    for oid, oside, osc in entry_cands:
                        if oid == pid or oid in open_pos:
                            continue
                        og = by_pair[oid]
                        olegs = pair_tickers(
                            oid, str(og["ticker_y"].iloc[0]), str(og["ticker_x"].iloc[0])
                        )
                        if shares_leg(legs, olegs) and (osc, oid) > (sc, pid):
                            blocked = True
                            break
            if cfg.corr_k is not None and not blocked:
                blocked = corr_blocks_candidate(
                    pid,
                    sc,
                    open_ids=list(open_pos.keys()),
                    same_bar_candidates=[(c[0], c[2]) for c in entry_cands],
                    abs_rho=rho_now,
                    k=float(cfg.corr_k),
                )
            if blocked:
                continue
            i = int(g.index[pd.to_datetime(g["date"]) == sig_date][0])
            fill_date = pd.Timestamp(g["date"].iloc[i + 1])
            profile = resolve_cost_profile(pid, ty, tx, cost_profile=cfg.cost_profile)
            by = leg_cost_bps(profile, ty, float(g["open_y"].iloc[i + 1]))
            bx = leg_cost_bps(profile, tx, float(g["open_x"].iloc[i + 1]))
            score = _score_series(g, cfg)[i]
            adf = float(g["adf_pvalue"].iloc[i]) if "adf_pvalue" in g.columns else float("nan")
            atr = float(g["atr_spread"].iloc[i]) if "atr_spread" in g.columns else float("nan")
            so = float(g["spread_open"].iloc[i + 1]) if "spread_open" in g.columns else float("nan")
            beta_fill = float(g[cfg.beta_column].iloc[i + 1])
            scale = pair_scale_from_score(
                score,
                adf,
                size_mode=cfg.size_mode,
                mean_abs_score=_mean_abs_at_entry(g, i, mean_abs_score),
            )
            size_mult = 1.0
            if cfg.exit_mode == "hl3_atr_breaker":
                size_mult = atr_size_multiplier(
                    atr=atr,
                    beta=beta_fill,
                    n_pairs=n_pairs,
                    pair_scale=scale,
                    leverage=1.0,
                    risk_frac=cfg.atr_risk_frac,
                )
            stop_spread = float("nan")
            if np.isfinite(so) and np.isfinite(atr) and atr > 0:
                stop_spread = float(so - side * atr)
            open_pos[pid] = {
                "side": side,
                "entry_idx": i + 1,
                "entry_date": fill_date,
                "entry_cost_bps": float(by + bx),
                "hl_at_entry": float(g["half_life"].iloc[i]) if "half_life" in g.columns else float("nan"),
                "cum_pnl": 0.0,
                "scale": float(scale) * float(size_mult),
                "stop_spread": stop_spread,
            }
            n_entries[pid] += 1
            pnl[pid][fill_date] -= (by + bx) / 10_000.0

        # Holding PnL for positions open through this signal's fill bar.
        for pid, st in list(open_pos.items()):
            g = by_pair[pid]
            rows = g.index[pd.to_datetime(g["date"]) == sig_date]
            if len(rows) == 0:
                continue
            i = int(rows[0])
            if i >= len(g) - 2:
                continue
            fill_date = pd.Timestamp(g["date"].iloc[i + 1])
            # Only accrue PnL if already open at this fill (entered on a prior bar).
            if pd.Timestamp(st["entry_date"]) > fill_date:
                continue
            if pd.Timestamp(st["entry_date"]) == fill_date and st["entry_idx"] == i + 1:
                # entered this bar: still accrue first hold open t+1 → t+2
                pass
            beta_fill = float(g[cfg.beta_column].iloc[i + 1])
            oy1, oy2 = float(g["open_y"].iloc[i + 1]), float(g["open_y"].iloc[i + 2])
            ox1, ox2 = float(g["open_x"].iloc[i + 1]), float(g["open_x"].iloc[i + 2])
            pos = int(st["side"])
            y_w = float(pos)
            x_w = -float(pos) * beta_fill if cfg.use_hedge_ratio_sizing else -float(pos)
            gross = abs(y_w) + abs(x_w)
            if gross > 0:
                y_w /= gross
                x_w /= gross
            bar_ret = (y_w * (oy2 / oy1 - 1.0) + x_w * (ox2 / ox1 - 1.0)) * float(st.get("scale", 1.0))
            bar_ret += daily_borrow_return(
                resolve_cost_profile(pid, ty, tx, cost_profile=cfg.cost_profile),
                y_weight=y_w,
                x_weight=x_w,
                scale=float(st.get("scale", 1.0)),
            )
            st["cum_pnl"] = float(st.get("cum_pnl", 0.0)) + bar_ret
            pnl[pid][fill_date] += bar_ret

    pair_results: dict[str, PairSimResult] = {}
    parts: list[pd.Series] = []
    for pid, pmap in pnl.items():
        ser = pd.Series(pmap, dtype=float).sort_index()
        ser.name = pid
        st = open_pos.get(pid)
        pair_results[pid] = PairSimResult(
            pair_id=pid,
            returns=ser,
            trades=pd.DataFrame(trades[pid], columns=list(_TRADE_COLS)),
            n_entries=n_entries[pid],
            n_open_at_end=int(st is not None),
            open_entry_cost_bps=float(st["entry_cost_bps"]) if st is not None else 0.0,
        )
        if not ser.empty:
            parts.append(ser)
    if not parts:
        return BookSimResult(returns=pd.Series(dtype=float, name="ret"), pair_results=pair_results)
    wide = pd.concat(parts, axis=1).fillna(0.0)
    raw = wide.mean(axis=1).rename("ret")
    if cfg.vol_mode != "s1_vt":
        out, _ = _densify_book_returns(raw, panel)
        return BookSimResult(returns=out, pair_results=pair_results)
    vt_cfg = VolTargetConfig(
        enabled=True,
        target_ann_vol=cfg.vt_target_ann_vol,
        periods_per_year=float(periods_per_year),
        min_periods=max(13, int(cfg.sigma_window // 4)),
    )
    scaled = []
    lev_vals = []
    lev_hist: list[float] = []
    lev_index: list[pd.Timestamp] = []
    prev_lev = 1.0
    for t, r in raw.items():
        leverage = leverage_from_history(
            pd.Series(lev_hist, index=pd.DatetimeIndex(lev_index)),
            vt_cfg,
            prev_leverage=prev_lev,
        )
        scaled.append(float(r) * float(leverage))
        lev_vals.append(leverage)
        lev_hist.append(float(r))
        lev_index.append(pd.Timestamp(t))
        prev_lev = leverage
    out = pd.Series(scaled, index=raw.index, name="ret")
    out, lev_ser = _densify_book_returns(
        out,
        panel,
        leverage=pd.Series(lev_vals, index=raw.index, dtype=float),
    )
    return BookSimResult(
        returns=out,
        pair_results=pair_results,
        leverage=lev_ser,
    )
