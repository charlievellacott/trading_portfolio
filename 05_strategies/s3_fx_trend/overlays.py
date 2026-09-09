"""Carry / value overlays (gate and tilt) for the S3 FX trend book."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_pair(pair: str) -> str:
    return str(pair).strip().upper().replace("_", "").replace("/", "").replace("=X", "")


def _rate_diff_series(pair: str, rates_df: pd.DataFrame) -> pd.Series:
    """Base minus quote policy rate (annual decimal) aligned to ``rates_df`` index."""
    p = _normalize_pair(pair)
    if len(p) != 6:
        raise ValueError(f"expected 6-letter pair, got {pair!r}")
    base, quote = p[:3], p[3:]
    cols = {str(c).upper(): c for c in rates_df.columns}
    if base not in cols or quote not in cols:
        # Allow columns named like EUR_rate / USD
        alt = {
            str(c).upper().replace("_RATE", "").replace("RATE_", ""): c
            for c in rates_df.columns
        }
        cols.update(alt)
    if base not in cols or quote not in cols:
        return pd.Series(np.nan, index=pd.to_datetime(rates_df.index), dtype=float)
    b = pd.to_numeric(rates_df[cols[base]], errors="coerce").astype(float)
    q = pd.to_numeric(rates_df[cols[quote]], errors="coerce").astype(float)
    out = (b - q).copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index().rename("rate_diff")


def carry_signal_weekly(
    pair: str,
    rates_df: pd.DataFrame,
    refresh_day: str = "MON",
) -> pd.Series:
    """Signed carry (base − quote) held constant between weekly refresh days.

    ``refresh_day`` is a pandas weekday alias (default Monday). Signal on day ``t``
    uses the last known rate differential as of close ``t`` (ffill), then is
    sampled on the refresh weekday and forward-filled — no lookahead.
    """
    if rates_df is None or rates_df.empty:
        return pd.Series(dtype=float, name="carry")
    diff = _rate_diff_series(pair, rates_df).dropna()
    if diff.empty:
        return pd.Series(dtype=float, name="carry")
    # Daily ffill of the differential, then freeze on weekly samples.
    daily = diff.asfreq("D").ffill()
    # Sample at end-of-day on refresh weekdays; use last available value that day.
    rule = str(refresh_day).upper()
    if not rule.endswith("W-"):
        # e.g. MON → W-MON
        rule = f"W-{rule}" if not rule.startswith("W-") else rule
    weekly = daily.resample(rule).last().dropna()
    # Map weekly samples back onto the daily calendar via backward asof (PIT).
    out = weekly.reindex(daily.index).ffill()
    out.name = "carry"
    return out


def value_signal_monthly(
    pair: str,
    reer_df: pd.DataFrame,
    refresh: str = "ME",
) -> pd.Series:
    """Monthly (or quarterly) REER value signal via ``pair_reer_value_signal``.

    Higher relative REER for the base ⇒ short-base lean (negative signal).
    Refresh freezes the last known PIT value onto the bar calendar.
    """
    if reer_df is None or reer_df.empty:
        return pd.Series(dtype=float, name="value")
    try:
        from data.ingestion.alternative_data.bis_reer import pair_reer_value_signal
    except ImportError:
        try:
            from data.ingestion.alternative_data.fred_fetcher import (
                pair_reer_value_signal,
            )
        except ImportError:
            # Fallback when BIS/FRED helpers are not yet on this branch.
            pair_reer_value_signal = _pair_reer_value_signal_fallback

    raw = pair_reer_value_signal(pair, reer_df)
    if raw is None or (isinstance(raw, pd.Series) and raw.empty):
        return pd.Series(dtype=float, name="value")
    s = pd.to_numeric(raw, errors="coerce").astype(float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().dropna()
    if s.empty:
        return pd.Series(dtype=float, name="value")
    how = str(refresh).lower()
    if how in {"quarterly", "qe", "q"}:
        sampled = s.resample("QE").last().dropna()
    else:
        # monthly / month-end
        rule = "ME" if str(refresh).upper() in {"ME", "M", "MONTHLY"} else str(refresh)
        sampled = s.resample(rule).last().dropna()
    out = sampled.reindex(s.index.union(sampled.index)).sort_index().ffill()
    out.name = "value"
    return out


def _pair_reer_value_signal_fallback(pair: str, reer_df: pd.DataFrame) -> pd.Series:
    """Local REER relative-value signal when fred_fetcher is unavailable.

    ``signal = -(log(REER_base) - log(REER_quote))`` z-scored over trailing
    available months (min 12). Expects wide monthly REER with currency columns
    and optional ``availability_date`` index (already PIT-lagged by caller).
    """
    p = _normalize_pair(pair)
    base, quote = p[:3], p[3:]
    cols = {str(c).upper(): c for c in reer_df.columns}
    if base not in cols or quote not in cols:
        return pd.Series(dtype=float, name="value")
    idx = pd.to_datetime(reer_df.index)
    b = pd.to_numeric(reer_df[cols[base]], errors="coerce").astype(float)
    q = pd.to_numeric(reer_df[cols[quote]], errors="coerce").astype(float)
    b.index = idx
    q.index = idx
    rel = np.log(b.clip(lower=1e-12)) - np.log(q.clip(lower=1e-12))
    # Trailing z of relative REER; negative of z so rich base → short.
    mu = rel.rolling(36, min_periods=12).mean()
    sd = rel.rolling(36, min_periods=12).std(ddof=1)
    z = (rel - mu) / sd.replace(0.0, np.nan)
    out = (-z).rename("value")
    return out.dropna()


def apply_overlay_gate(
    base_signal: pd.Series,
    overlay_signal: pd.Series,
) -> pd.Series:
    """Keep ``base_signal`` only when overlay agrees in sign; else flat."""
    base = pd.to_numeric(base_signal, errors="coerce").astype(float)
    ov = pd.to_numeric(overlay_signal, errors="coerce").astype(float)
    ov = ov.reindex(base.index).ffill()
    agree = np.sign(base) * np.sign(ov) > 0
    # Missing overlay → allow (no gate information).
    allow = agree | ~np.isfinite(ov.to_numpy(dtype=float)) | (ov == 0)
    out = base.where(allow, 0.0)
    out.name = base_signal.name
    return out


def apply_overlay_tilt(
    base_weight: pd.Series,
    overlay_signal: pd.Series,
    agree_scale: float,
    disagree_scale: float,
) -> pd.Series:
    """Scale weights up/down when overlay agrees / disagrees with weight sign."""
    w = pd.to_numeric(base_weight, errors="coerce").astype(float)
    ov = pd.to_numeric(overlay_signal, errors="coerce").astype(float)
    ov = ov.reindex(w.index).ffill()
    agree = np.sign(w) * np.sign(ov) > 0
    disagree = np.sign(w) * np.sign(ov) < 0
    scale = pd.Series(1.0, index=w.index, dtype=float)
    scale = scale.where(~agree, float(agree_scale))
    scale = scale.where(~disagree, float(disagree_scale))
    # Zero / missing overlay → scale 1.
    missing = ~np.isfinite(ov.to_numpy(dtype=float)) | (ov == 0)
    scale = scale.where(~missing, 1.0)
    out = (w * scale).astype(float)
    out.name = base_weight.name
    return out
