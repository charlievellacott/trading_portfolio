"""Map entry-date weights onto hold intervals for each timing mode."""

from __future__ import annotations

import pandas as pd

from backtest.s1_equities.signals import (
    TIMING_MON_OPEN_FRI_CLOSE,
    TIMING_MON_OPEN_MON_OPEN,
    VALID_TIMING_MODES,
)


def _iso_year_week(ts: pd.Timestamp) -> tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


def exit_date_for_entry(
    entry: pd.Timestamp,
    timing_mode: str,
    *,
    entry_dates: pd.DatetimeIndex,
    trading_calendar: pd.DatetimeIndex,
) -> pd.Timestamp | pd.NaT:
    """
    Exit timestamp for a Monday (week-start) entry under ``timing_mode``.

    - ``mon_open_mon_open``: next entry date (next week-start open).
    - ``mon_open_fri_close``: last trading day in the same ISO week with
      weekday <= Friday (typically Friday close).
    """
    if timing_mode not in VALID_TIMING_MODES:
        raise ValueError(f"unknown timing_mode={timing_mode!r}")

    entry = pd.Timestamp(entry)
    cal = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).sort_values().unique()
    entries = pd.DatetimeIndex(pd.to_datetime(entry_dates)).sort_values().unique()

    if timing_mode == TIMING_MON_OPEN_MON_OPEN:
        pos = entries.get_indexer([entry], method=None)[0]
        if pos < 0 or pos + 1 >= len(entries):
            return pd.NaT
        return entries[pos + 1]

    # Friday close (or last session Mon–Fri in that ISO week)
    y, w = _iso_year_week(entry)
    week_days = [
        d
        for d in cal
        if _iso_year_week(pd.Timestamp(d)) == (y, w) and pd.Timestamp(d).weekday() <= 4
    ]
    if not week_days:
        return pd.NaT
    # Prefer Friday if present; else last trading day of the week
    fridays = [d for d in week_days if pd.Timestamp(d).weekday() == 4]
    return pd.Timestamp(fridays[-1] if fridays else week_days[-1])


def build_hold_table(
    entry_weights: pd.DataFrame,
    timing_mode: str,
    *,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    One row per entry date with exit date for the chosen timing mode.

    Columns: ``entry_date``, ``exit_date``.
    """
    entries = entry_weights.index
    rows = []
    for dt in entries:
        if float(entry_weights.loc[dt].abs().sum()) < 1e-12:
            continue
        ex = exit_date_for_entry(
            dt,
            timing_mode,
            entry_dates=entries,
            trading_calendar=trading_calendar,
        )
        if pd.isna(ex):
            continue
        rows.append({"entry_date": pd.Timestamp(dt), "exit_date": pd.Timestamp(ex)})
    return pd.DataFrame(rows)
