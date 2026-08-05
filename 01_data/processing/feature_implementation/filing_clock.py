"""H-010 supporting: SEC filing event clock primitives and panel helpers."""

from __future__ import annotations

import pandas as pd

from data.processing.feature_implementation.utilities import _require_columns


# ---------------------------------------------------------------------------
# Series primitives
# ---------------------------------------------------------------------------


def days_since_filing(
    date: pd.Series,
    last_filed: pd.Series,
) -> pd.Series:
    """Calendar days since ``last_filed``; NaN when either side is missing."""
    d = pd.to_datetime(date, errors="coerce")
    filed = pd.to_datetime(last_filed, errors="coerce")
    delta = d - filed
    return pd.Series(delta.dt.days, index=date.index, dtype=float)


def expected_days_until_filing(
    date: pd.Series,
    expected_next_filed: pd.Series,
) -> pd.Series:
    """
    Signed calendar days until ``expected_next_filed``.

    Negative when ``date`` is past the expected filing (overdue). NaN when
    either side is missing.
    """
    d = pd.to_datetime(date, errors="coerce")
    nxt = pd.to_datetime(expected_next_filed, errors="coerce")
    delta = nxt - d
    return pd.Series(delta.dt.days, index=date.index, dtype=float)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def add_days_since_filing(
    panel: pd.DataFrame,
    *,
    col: str = "days_since_filing",
) -> pd.DataFrame:
    """Return a copy with calendar days since ``last_filed`` in ``col``."""
    _require_columns(panel, {"date", "ticker", "last_filed"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    result = panel.copy()
    result[col] = days_since_filing(result["date"], result["last_filed"])
    return result


def add_expected_days_until_filing(
    panel: pd.DataFrame,
    *,
    col: str = "expected_days_until_filing",
) -> pd.DataFrame:
    """Return a copy with signed days until ``expected_next_filed`` in ``col``."""
    _require_columns(panel, {"date", "ticker", "expected_next_filed"})
    if panel.empty:
        out = panel.copy()
        out[col] = pd.Series(dtype=float)
        return out
    result = panel.copy()
    result[col] = expected_days_until_filing(
        result["date"], result["expected_next_filed"]
    )
    return result
