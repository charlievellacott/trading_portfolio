"""Expand S1 weekly sealed returns onto weekdays for FTMO daily rules."""

from __future__ import annotations

import pandas as pd


def weekly_to_weekday_returns(weekly: pd.Series) -> pd.Series:
    """Place each weekly simple return on its index date; other weekdays = 0.

    Does **not** divide the week by 5. Compounding Mon–Fri recovers the weekly
    return. Intended approximation: S1 decides Monday; intra-week MTM is
    unknown, so putting the full week PnL on the decision day is conservative
    for the 5% daily-loss rule. About one FTMO trading day is counted per week
    (nonzero Monday).
    """
    s = pd.to_numeric(weekly, errors="coerce").astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    if s.empty:
        return pd.Series(dtype=float, name=weekly.name)
    pieces: list[pd.Series] = []
    for ts, ret in s.items():
        t0 = pd.Timestamp(ts).normalize()
        week_end = t0 + pd.offsets.BDay(4)
        days = pd.bdate_range(t0, week_end, freq="B")
        if len(days) == 0:
            continue
        vals = [float(ret)] + [0.0] * (len(days) - 1)
        # If t0 is not a weekday, still put the return on t0 only.
        if t0.weekday() >= 5:
            pieces.append(pd.Series([float(ret)], index=pd.DatetimeIndex([t0])))
            continue
        pieces.append(pd.Series(vals, index=days))
    out = pd.concat(pieces)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.name = s.name if s.name is not None else "ret"
    return out.astype(float)
