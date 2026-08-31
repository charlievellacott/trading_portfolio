"""Live book walk: engine entry/exit rules with a last-bar close-t decision.

Research ``simulate_book`` stops two bars early because it needs open ``t+1`` and
open ``t+2`` for fill + first PnL. Live needs the signal at the last completed
close ``t`` (fill at the next open, β known at close ``t``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from risk.analytics.s1_equities.vol_targeting import VolTargetConfig, leverage_from_history
from strategies.s2_coint.baseline import PERIODS_PER_YEAR
from strategies.s2_coint.config import S2SimConfig
from strategies.s2_coint.costs import daily_borrow_return, resolve_cost_profile
from strategies.s2_coint.engine import (
    _health_blocks_entry,
    _health_flattens,
    _hmm_blocks,
    _k_in,
    _k_out,
    _score_series,
    _trend_blocks,
)
from strategies.s2_coint.overlap import pair_tickers, score_confidence, shares_leg
from strategies.s2_coint.sizing import (
    gross_normalized_legs,
    pair_scale_from_score,
    rolling_mean_abs_score,
)


@dataclass
class LiveBookResult:
    """Target book after the last completed close (signal ``t``)."""

    signal_date: pd.Timestamp
    weights: pd.DataFrame
    open_pos: dict[str, dict] = field(default_factory=dict)
    unlevered_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    leverage: float = 1.0
    pair_diag: pd.DataFrame = field(default_factory=pd.DataFrame)


def attach_rolling_mean_abs_score(
    panel: pd.DataFrame,
    window: int,
    *,
    score_column: str = "z",
) -> pd.DataFrame:
    """Per-pair PIT rolling mean of ``|score|`` (window = ``Z_WINDOW_STAR`` in live)."""
    if panel is None or panel.empty:
        return panel
    parts: list[pd.DataFrame] = []
    for _, g in panel.groupby("pair_id", sort=False):
        g = g.sort_values("date").copy()
        g["mean_abs_score"] = rolling_mean_abs_score(g[score_column], window)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _vol_cfg(cfg: S2SimConfig, periods_per_year: float) -> VolTargetConfig:
    return VolTargetConfig(
        enabled=cfg.vol_mode == "s1_vt",
        target_ann_vol=cfg.vt_target_ann_vol,
        periods_per_year=float(periods_per_year),
        min_periods=max(13, int(cfg.sigma_window // 4)),
    )


def _row_at(g: pd.DataFrame, sig_date: pd.Timestamp) -> int | None:
    rows = g.index[pd.to_datetime(g["date"]) == pd.Timestamp(sig_date)]
    if len(rows) == 0:
        return None
    return int(rows[0])


def _sigma_at(g: pd.DataFrame, i: int, cfg: S2SimConfig) -> float:
    if "spread" not in g.columns:
        return float("nan")
    s = g["spread"].astype(float)
    if i + 1 < cfg.sigma_window:
        return float("nan")
    return float(s.iloc[i - cfg.sigma_window + 1 : i + 1].std(ddof=1))


def _pair_scalar(g: pd.DataFrame, i: int, col: str, default: float = float("nan")) -> float:
    if col not in g.columns:
        return float(default)
    return float(g[col].iloc[i])


def walk_live_book(
    panel: pd.DataFrame,
    cfg: S2SimConfig | None = None,
    *,
    asof: pd.Timestamp | str | None = None,
    universe_tickers: list[str] | None = None,
) -> LiveBookResult:
    """Replay STAR rules through ``asof`` (inclusive) and emit target weights.

    Historical bars with two future opens accrue unlevered book returns for
    ``s1_vt`` (same PIT as the engine: L on date t uses returns strictly before
    t). The last close has no next open: decisions use β / z / ADF on that row.
    """
    cfg = cfg or S2SimConfig()
    if panel is None or panel.empty:
        return LiveBookResult(
            signal_date=pd.NaT,
            weights=_empty_weights(universe_tickers or []),
        )

    d = panel.copy()
    d["date"] = pd.to_datetime(d["date"])
    if asof is not None:
        cutoff = pd.Timestamp(asof).normalize()
        d = d.loc[d["date"] <= cutoff].copy()
    if d.empty:
        return LiveBookResult(
            signal_date=pd.NaT,
            weights=_empty_weights(universe_tickers or []),
        )

    score_col = cfg.score_column if cfg.score_column in d.columns else "z"
    d = attach_rolling_mean_abs_score(d, cfg.z_window, score_column=score_col)

    by_pair = {
        str(pid): g.sort_values("date").reset_index(drop=True)
        for pid, g in d.groupby("pair_id", sort=False)
    }
    n_pairs = max(len(by_pair), 1)
    signal_dates = sorted({pd.Timestamp(t) for g in by_pair.values() for t in g["date"]})
    open_pos: dict[str, dict] = {}
    pnl: dict[str, dict[pd.Timestamp, float]] = {pid: defaultdict(float) for pid in by_pair}

    for sig_date in signal_dates:
        exit_ids: list[str] = []
        entry_cands: list[tuple[str, int, float]] = []
        for pid, g in by_pair.items():
            i = _row_at(g, sig_date)
            if i is None:
                continue
            score = float(_score_series(g, cfg)[i])
            adf = _pair_scalar(g, i, "adf_pvalue")
            vj = _pair_scalar(g, i, "variance_jump")
            hl = _pair_scalar(g, i, "half_life")
            rsi = _pair_scalar(g, i, "rsi_spread")
            adx = _pair_scalar(g, i, "adx_spread")
            kin = _k_in(cfg, _sigma_at(g, i, cfg))
            kout = _k_out(cfg)
            st = open_pos.get(pid)
            if st is not None:
                pos = int(st["side"])
                flatten = np.isfinite(score) and (pos * score >= kout)
                flatten = flatten or _health_flattens(adf, vj, cfg)
                if flatten:
                    exit_ids.append(pid)
            else:
                want_long = np.isfinite(score) and np.isfinite(kin) and score <= -kin
                want_short = np.isfinite(score) and np.isfinite(kin) and score >= kin
                side = 1 if want_long else (-1 if want_short else 0)
                if (
                    side != 0
                    and not _health_blocks_entry(adf, vj, hl, cfg)
                    and not _trend_blocks(side, rsi, adx, cfg)
                    and not _hmm_blocks(g, i, cfg)
                ):
                    entry_cands.append((pid, side, score_confidence(score, adf)))

        for pid in exit_ids:
            open_pos.pop(pid, None)

        if cfg.overlap_mode == "never_allow":
            entry_cands.sort(key=lambda t: (t[2], t[0]), reverse=True)

        for pid, side, sc in entry_cands:
            g = by_pair[pid]
            i = _row_at(g, sig_date)
            if i is None:
                continue
            if i + 1 < len(g):
                beta_fill = float(g[cfg.beta_column].iloc[i + 1])
                fill_date = pd.Timestamp(g["date"].iloc[i + 1])
            else:
                beta_fill = float(g[cfg.beta_column].iloc[i])
                fill_date = pd.Timestamp(g["date"].iloc[i])
            if not np.isfinite(beta_fill):
                continue
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
            if blocked:
                continue
            score = float(_score_series(g, cfg)[i])
            adf = _pair_scalar(g, i, "adf_pvalue")
            mean_abs = _pair_scalar(g, i, "mean_abs_score", default=1.0)
            scale = pair_scale_from_score(
                score, adf, size_mode=cfg.size_mode, mean_abs_score=mean_abs
            )
            open_pos[pid] = {
                "side": int(side),
                "entry_idx": i + 1 if i + 1 < len(g) else i,
                "entry_date": fill_date,
                "scale": float(scale),
                "mean_abs_score": float(mean_abs) if np.isfinite(mean_abs) else 1.0,
                "z_at_entry": float(score),
            }

        for pid, st in list(open_pos.items()):
            g = by_pair[pid]
            i = _row_at(g, sig_date)
            if i is None or i + 2 >= len(g):
                continue
            fill_date = pd.Timestamp(g["date"].iloc[i + 1])
            if pd.Timestamp(st["entry_date"]) > fill_date:
                continue
            beta_fill = float(g[cfg.beta_column].iloc[i + 1])
            if not np.isfinite(beta_fill):
                continue
            oy1 = float(g["open_y"].iloc[i + 1])
            oy2 = float(g["open_y"].iloc[i + 2])
            ox1 = float(g["open_x"].iloc[i + 1])
            ox2 = float(g["open_x"].iloc[i + 2])
            if min(oy1, oy2, ox1, ox2) <= 0:
                continue
            pos = int(st["side"])
            y_w, x_w = gross_normalized_legs(pos, beta_fill)
            bar_ret = (y_w * (oy2 / oy1 - 1.0) + x_w * (ox2 / ox1 - 1.0)) * float(
                st.get("scale", 1.0)
            )
            ty, tx = str(g["ticker_y"].iloc[0]), str(g["ticker_x"].iloc[0])
            bar_ret += daily_borrow_return(
                resolve_cost_profile(pid, ty, tx, cost_profile=cfg.cost_profile),
                y_weight=y_w,
                x_weight=x_w,
                scale=float(st.get("scale", 1.0)),
            )
            pnl[pid][fill_date] += bar_ret

    signal_date = signal_dates[-1]
    unlev = _mean_book_returns(pnl, n_pairs)
    vt_cfg = _vol_cfg(cfg, PERIODS_PER_YEAR)
    prev_lev = 1.0
    lev_hist: list[float] = []
    lev_index: list[pd.Timestamp] = []
    for t, r in unlev.items():
        leverage = leverage_from_history(
            pd.Series(lev_hist, index=pd.DatetimeIndex(lev_index)),
            vt_cfg,
            prev_leverage=prev_lev,
        )
        lev_hist.append(float(r))
        lev_index.append(pd.Timestamp(t))
        prev_lev = leverage
    leverage = leverage_from_history(
        pd.Series(lev_hist, index=pd.DatetimeIndex(lev_index)),
        vt_cfg,
        prev_leverage=prev_lev,
    )

    weights = weights_from_open_pos(
        open_pos,
        by_pair,
        cfg,
        signal_date=signal_date,
        leverage=float(leverage),
        n_pairs=n_pairs,
        universe_tickers=universe_tickers,
    )
    diag = _pair_diag_frame(by_pair, open_pos, signal_date, cfg)
    return LiveBookResult(
        signal_date=signal_date,
        weights=weights,
        open_pos=open_pos,
        unlevered_returns=unlev,
        leverage=float(leverage),
        pair_diag=diag,
    )


def _mean_book_returns(
    pnl: dict[str, dict[pd.Timestamp, float]],
    n_pairs: int,
) -> pd.Series:
    parts: list[pd.Series] = []
    for pid, pmap in pnl.items():
        ser = pd.Series(pmap, dtype=float).sort_index()
        ser.name = pid
        parts.append(ser)
    if not parts:
        return pd.Series(dtype=float, name="ret")
    wide = pd.concat(parts, axis=1).fillna(0.0)
    n = max(int(n_pairs), 1)
    # Always divide by book size (engine averages the pair columns it emitted).
    raw = wide.sum(axis=1) / float(n)
    raw.name = "ret"
    return raw.sort_index()


def _empty_weights(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        rows.append(
            {
                "ticker": t,
                "date": pd.NaT,
                "pair_id": "",
                "side": 0,
                "z": float("nan"),
                "score": float("nan"),
                "beta": float("nan"),
                "mean_abs_score": float("nan"),
                "scale": 0.0,
                "leverage": 1.0,
                "weight": 0.0,
                "close": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def weights_from_open_pos(
    open_pos: dict[str, dict],
    by_pair: dict[str, pd.DataFrame],
    cfg: S2SimConfig,
    *,
    signal_date: pd.Timestamp,
    leverage: float,
    n_pairs: int,
    universe_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Map open pairs to per-ticker signed weights (gross-normalised × scale/n × L)."""
    n = max(int(n_pairs), 1)
    rows: list[dict] = []
    seen: set[str] = set()
    for pid, st in open_pos.items():
        g = by_pair[pid]
        i = _row_at(g, signal_date)
        last = g.iloc[i] if i is not None else g.iloc[-1]
        beta = float(last[cfg.beta_column])
        if not np.isfinite(beta):
            continue
        side = int(st["side"])
        scale = float(st.get("scale", 1.0))
        y_w, x_w = gross_normalized_legs(side, beta)
        factor = (scale / float(n)) * float(leverage)
        ty = str(last["ticker_y"])
        tx = str(last["ticker_x"])
        z = float(last["z"]) if "z" in last.index else float("nan")
        mean_abs = float(st.get("mean_abs_score", 1.0))
        if not np.isfinite(mean_abs) and "mean_abs_score" in last.index:
            mean_abs = float(last["mean_abs_score"])
        for ticker, w_leg, px in (
            (ty, y_w, float(last["close_y"])),
            (tx, x_w, float(last["close_x"])),
        ):
            rows.append(
                {
                    "ticker": ticker,
                    "date": pd.Timestamp(signal_date),
                    "pair_id": pid,
                    "side": side,
                    "z": z,
                    "score": z,
                    "beta": beta,
                    "mean_abs_score": mean_abs,
                    "scale": scale,
                    "leverage": float(leverage),
                    "weight": float(w_leg) * float(factor),
                    "close": px,
                }
            )

    all_tickers = list(universe_tickers or [])
    if not all_tickers:
        all_tickers = []
        seen_u: set[str] = set()
        for g in by_pair.values():
            ty = str(g["ticker_y"].iloc[0])
            tx = str(g["ticker_x"].iloc[0])
            for t in (ty, tx):
                if t not in seen_u:
                    seen_u.add(t)
                    all_tickers.append(t)
    by_ticker = {str(r["ticker"]): r for r in rows}
    out_rows = []
    for t in all_tickers:
        if t in by_ticker:
            out_rows.append(by_ticker[t])
        else:
            out_rows.append(
                {
                    "ticker": t,
                    "date": pd.Timestamp(signal_date),
                    "pair_id": "",
                    "side": 0,
                    "z": float("nan"),
                    "score": float("nan"),
                    "beta": float("nan"),
                    "mean_abs_score": float("nan"),
                    "scale": 0.0,
                    "leverage": float(leverage),
                    "weight": 0.0,
                    "close": float("nan"),
                }
            )
    return pd.DataFrame(out_rows)


def _pair_diag_frame(
    by_pair: dict[str, pd.DataFrame],
    open_pos: dict[str, dict],
    signal_date: pd.Timestamp,
    cfg: S2SimConfig,
) -> pd.DataFrame:
    rows = []
    for pid, g in by_pair.items():
        i = _row_at(g, signal_date)
        if i is None:
            continue
        last = g.iloc[i]
        st = open_pos.get(pid)
        rows.append(
            {
                "pair_id": pid,
                "date": pd.Timestamp(signal_date),
                "z": float(last["z"]) if "z" in last.index else float("nan"),
                "beta": float(last[cfg.beta_column]),
                "adf_pvalue": _pair_scalar(g, i, "adf_pvalue"),
                "variance_jump": _pair_scalar(g, i, "variance_jump"),
                "mean_abs_score": _pair_scalar(g, i, "mean_abs_score", default=1.0),
                "side": int(st["side"]) if st is not None else 0,
                "scale": float(st["scale"]) if st is not None else 0.0,
            }
        )
    return pd.DataFrame(rows)
