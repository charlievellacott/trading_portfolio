"""PIT spread OHLC envelope from close-t α/β (never feed H/L into the hedge)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.processing.feature_implementation.atr import wilder_atr


def spread_ohlc_frame(g: pd.DataFrame) -> pd.DataFrame:
    """Log-spread open/high/low/close using close-t alpha/beta.

    For ``beta > 0``: high envelope uses ``high_y`` vs ``low_x``; low envelope
    uses ``low_y`` vs ``high_x``. Beta/alpha are from closes, not H/L.
    """
    d = g.sort_values("date")
    alpha = d["alpha"].to_numpy(dtype=float)
    beta = d["beta"].to_numpy(dtype=float)
    oy = np.log(np.clip(d["open_y"].to_numpy(dtype=float), 1e-12, None))
    ox = np.log(np.clip(d["open_x"].to_numpy(dtype=float), 1e-12, None))
    hy = np.log(np.clip(d["high_y"].to_numpy(dtype=float), 1e-12, None))
    hx = np.log(np.clip(d["high_x"].to_numpy(dtype=float), 1e-12, None))
    ly = np.log(np.clip(d["low_y"].to_numpy(dtype=float), 1e-12, None))
    lx = np.log(np.clip(d["low_x"].to_numpy(dtype=float), 1e-12, None))
    cy = np.log(np.clip(d["close_y"].to_numpy(dtype=float), 1e-12, None))
    cx = np.log(np.clip(d["close_x"].to_numpy(dtype=float), 1e-12, None))

    s_open = oy - alpha - beta * ox
    s_close = cy - alpha - beta * cx
    # Conservative range: β>0 → y high / x low is the high spread corner.
    s_high = np.where(beta >= 0.0, hy - alpha - beta * lx, hy - alpha - beta * hx)
    s_low = np.where(beta >= 0.0, ly - alpha - beta * hx, ly - alpha - beta * lx)
    s_high = np.maximum(s_high, np.maximum(s_open, s_close))
    s_low = np.minimum(s_low, np.minimum(s_open, s_close))
    out = pd.DataFrame(
        {
            "spread_open": s_open,
            "spread_high": s_high,
            "spread_low": s_low,
            "spread_close": s_close,
        },
        index=d.index,
    )
    return out


def attach_spread_indicators(
    panel: pd.DataFrame,
    *,
    rsi_period: int = 14,
    adx_period: int = 14,
    atr_window: int = 14,
    include_rsi_adx: bool = True,
) -> pd.DataFrame:
    """Add spread OHLC and Wilder ATR. RSI/ADX need TA-Lib (H-005)."""
    if panel.empty:
        return panel.copy()
    parts: list[pd.DataFrame] = []
    rsi_fn = adx_fn = None
    if include_rsi_adx:
        from data.processing.feature_implementation.talib_features import adx_series, rsi_series

        rsi_fn, adx_fn = rsi_series, adx_series
    for _, g in panel.groupby("pair_id", sort=False):
        g = g.sort_values("date").copy()
        ohlc = spread_ohlc_frame(g)
        for col in ohlc.columns:
            g[col] = ohlc[col].to_numpy(dtype=float)
        if rsi_fn is not None:
            g["rsi_spread"] = rsi_fn(g["spread_close"], timeperiod=rsi_period).to_numpy(
                dtype=float
            )
            g["adx_spread"] = adx_fn(
                g["spread_high"],
                g["spread_low"],
                g["spread_close"],
                timeperiod=adx_period,
            ).to_numpy(dtype=float)
        # S2 close-t decision may use close-t ATR (fill is next open).
        g["atr_spread"] = wilder_atr(
            g["spread_high"],
            g["spread_low"],
            g["spread_close"],
            window=atr_window,
            pit_shift=0,
        ).to_numpy(dtype=float)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
