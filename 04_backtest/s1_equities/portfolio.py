"""Cross-sectional top/bottom-N portfolio weights from scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.realized_vol import (
    DEFAULT_OPEN_VOL_WINDOW,
)

WEIGHT_EQUAL = "equal"
WEIGHT_SCORE = "score"
WEIGHT_RANK = "rank"
WEIGHT_INV_VOL = "inv_vol"

VALID_WEIGHT_MODES = frozenset(
    {WEIGHT_EQUAL, WEIGHT_SCORE, WEIGHT_RANK, WEIGHT_INV_VOL}
)

# Re-export so callers share one window constant with feature_implementation.
INV_VOL_WINDOW = DEFAULT_OPEN_VOL_WINDOW


def _select_top_bottom(
    row: pd.Series,
    n: int,
    *,
    need: int,
) -> tuple[pd.Index, pd.Index] | None:
    """Return (top, bot) index sets, or None if the date should be flat."""
    s = row.dropna()
    if len(s) < need:
        return None
    top = s.nlargest(n).index
    bot = s.nsmallest(n).index
    overlap = top.intersection(bot)
    if len(overlap):
        top = top.difference(overlap)
        bot = bot.difference(overlap)
        if len(top) == 0 or len(bot) == 0:
            return None
    return top, bot


def _sleeve_raw_weights(
    names: pd.Index,
    *,
    weight_mode: str,
    scores_row: pd.Series,
    vol_row: pd.Series | None,
    long_side: bool,
) -> pd.Series:
    """Non-negative raw weights within one sleeve (before sleeve renormalize)."""
    if weight_mode == WEIGHT_EQUAL:
        return pd.Series(1.0, index=names, dtype=float)

    if weight_mode == WEIGHT_SCORE:
        raw = scores_row.reindex(names).astype(float)
        raw = raw.abs() if not long_side else raw
        raw = raw.where(np.isfinite(raw) & (raw > 0))
        return raw

    if weight_mode == WEIGHT_RANK:
        sc = scores_row.reindex(names).astype(float)
        # Higher score → higher long weight; lower score → higher short weight.
        if long_side:
            return sc.rank(method="average", ascending=True)
        return (-sc).rank(method="average", ascending=True)

    if weight_mode == WEIGHT_INV_VOL:
        if vol_row is None:
            raise ValueError("vol is required for weight_mode='inv_vol'")
        v = vol_row.reindex(names).astype(float)
        raw = (1.0 / v).where(np.isfinite(v) & (v > 0))
        return raw

    raise ValueError(f"unknown weight_mode={weight_mode!r}")


def _normalize_sleeve(raw: pd.Series, *, signed_sum: float) -> pd.Series:
    """Scale finite positive raw weights so they sum to ``signed_sum``."""
    w = raw.replace([np.inf, -np.inf], np.nan).dropna()
    w = w.where(w > 0).dropna()
    if w.empty:
        return w
    total = float(w.sum())
    if total <= 0 or not np.isfinite(total):
        return pd.Series(dtype=float)
    return w * (signed_sum / total)


def build_entry_weights(
    scores: pd.DataFrame,
    n: int,
    *,
    weight_mode: str = WEIGHT_EQUAL,
    vol: pd.DataFrame | None = None,
    min_names: int | None = None,
) -> pd.DataFrame:
    """
    Dollar-neutral long top-N / short bottom-N weights per date.

    Each sleeve is renormalized to ``±0.5`` after applying ``weight_mode``.
    Names with missing inputs for the chosen mode are dropped from that sleeve
    before renormalization; empty sleeves yield a flat (all-zero) row.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if weight_mode not in VALID_WEIGHT_MODES:
        raise ValueError(f"unknown weight_mode={weight_mode!r}")
    if weight_mode == WEIGHT_INV_VOL and vol is None:
        raise ValueError("vol is required for weight_mode='inv_vol'")

    need = min_names if min_names is not None else 2 * n
    out = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for dt, row in scores.iterrows():
        selected = _select_top_bottom(row, n, need=need)
        if selected is None:
            continue
        top, bot = selected
        if vol is None:
            vol_row = None
        elif dt in vol.index:
            vol_row = vol.reindex(columns=scores.columns).loc[dt]
        else:
            vol_row = pd.Series(np.nan, index=scores.columns, dtype=float)

        long_raw = _sleeve_raw_weights(
            top,
            weight_mode=weight_mode,
            scores_row=row,
            vol_row=vol_row,
            long_side=True,
        )
        short_raw = _sleeve_raw_weights(
            bot,
            weight_mode=weight_mode,
            scores_row=row,
            vol_row=vol_row,
            long_side=False,
        )
        long_w = _normalize_sleeve(long_raw, signed_sum=0.5)
        short_w = _normalize_sleeve(short_raw, signed_sum=0.5)
        if long_w.empty or short_w.empty:
            continue
        out.loc[dt, long_w.index] = long_w.to_numpy()
        out.loc[dt, short_w.index] = (-short_w).to_numpy()

    return out


def top_bottom_weights(
    scores: pd.DataFrame,
    n: int,
    *,
    min_names: int | None = None,
) -> pd.DataFrame:
    """
    Equal-weight dollar-neutral long top-N / short bottom-N per date.

    Each long name gets ``+1/(2N)``, each short ``-1/(2N)`` when at least
    ``2N`` finite scores exist (or ``min_names`` if provided). Otherwise the
    row is all zeros.
    """
    return build_entry_weights(
        scores,
        n,
        weight_mode=WEIGHT_EQUAL,
        min_names=min_names,
    )


def gross_exposure(weights: pd.DataFrame) -> pd.Series:
    """Sum of absolute weights per date."""
    return weights.abs().sum(axis=1)


def net_exposure(weights: pd.DataFrame) -> pd.Series:
    """Sum of signed weights per date (≈ 0 if dollar-neutral)."""
    return weights.sum(axis=1)


def weight_concentration_stats(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Per-date concentration diagnostics for non-flat books.

    Columns include max/min/mean/median |w|, sleeve max shares of the ±0.5
    sleeve budget, and HHI (sum of w² over active names).
    """
    rows: list[dict] = []
    for dt, row in weights.iterrows():
        active = row.replace(0.0, np.nan).dropna()
        if active.empty:
            continue
        abs_w = active.abs()
        long = active[active > 0]
        short = active[active < 0]
        long_max_share = (
            float(long.max() / 0.5) if len(long) and float(long.sum()) != 0 else np.nan
        )
        short_max_share = (
            float((-short).max() / 0.5)
            if len(short) and float(short.sum()) != 0
            else np.nan
        )
        rows.append(
            {
                "date": dt,
                "n_long": int(len(long)),
                "n_short": int(len(short)),
                "max_abs_w": float(abs_w.max()),
                "min_abs_w": float(abs_w.min()),
                "mean_abs_w": float(abs_w.mean()),
                "p50_abs_w": float(abs_w.median()),
                "long_max_share": long_max_share,
                "short_max_share": short_max_share,
                "hhi": float((active.to_numpy(dtype=float) ** 2).sum()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "n_long",
                "n_short",
                "max_abs_w",
                "min_abs_w",
                "mean_abs_w",
                "p50_abs_w",
                "long_max_share",
                "short_max_share",
                "hhi",
            ]
        )
    out = pd.DataFrame(rows).set_index("date").sort_index()
    return out


def summarize_concentration(stats: pd.DataFrame) -> pd.Series:
    """
    Aggregate concentration stats across dates (mean / p50 / p95 / max of max_|w|).

    Useful for deciding a future per-name weight cap from observed extremes.
    """
    if stats is None or stats.empty or "max_abs_w" not in stats.columns:
        return pd.Series(
            {
                "n_dates": 0,
                "max_abs_w_mean": np.nan,
                "max_abs_w_p50": np.nan,
                "max_abs_w_p95": np.nan,
                "max_abs_w_max": np.nan,
                "min_abs_w_mean": np.nan,
                "hhi_mean": np.nan,
                "long_max_share_mean": np.nan,
                "long_max_share_max": np.nan,
                "short_max_share_mean": np.nan,
                "short_max_share_max": np.nan,
            },
            dtype=float,
        )
    m = stats["max_abs_w"]
    return pd.Series(
        {
            "n_dates": float(len(stats)),
            "max_abs_w_mean": float(m.mean()),
            "max_abs_w_p50": float(m.median()),
            "max_abs_w_p95": float(m.quantile(0.95)),
            "max_abs_w_max": float(m.max()),
            "min_abs_w_mean": float(stats["min_abs_w"].mean()),
            "hhi_mean": float(stats["hhi"].mean()),
            "long_max_share_mean": float(stats["long_max_share"].mean()),
            "long_max_share_max": float(stats["long_max_share"].max()),
            "short_max_share_mean": float(stats["short_max_share"].mean()),
            "short_max_share_max": float(stats["short_max_share"].max()),
        },
        dtype=float,
    )
