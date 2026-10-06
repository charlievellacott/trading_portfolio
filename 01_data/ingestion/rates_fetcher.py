"""
G10 policy / short-rate series via FRED for S3 FX carry / rate differentials.

FRED levels are percent (e.g. ``5.33`` = 5.33%). Public helpers return **annual
decimals** (``0.0533``) for swap / carry math in ``strategies.s3_fx_trend``.

``POLICY_RATE_SERIES`` maps ISO currency → FRED series id:

- USD: DFF (effective federal funds)
- EUR: ECBDFR (ECB deposit facility)
- GBP: IUDSOIA (SONIA / overnight sterling proxy)
- JPY: INTDSRJPM193N (discount rate Japan — verify for research use)
- CHF: IRSTCI01CHM156N (commonly cited; **verify** against OECD/BIS docs)
- CAD: IRSTCI01CAM156N (commonly cited; **verify**)
- AUD: IRSTCI01AUM156N (commonly cited; **verify**)
- NZD: IRSTCI01NZM156N (commonly cited; **verify**)

CHF / CAD / AUD / NZD FRED ids above are commonly cited in research notebooks
but should be verified before production use — series definitions and
revisions change.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from data.ingestion.alternative_data.fred_fetcher import fetch_fred_series

logger = logging.getLogger(__name__)

POLICY_RATE_SERIES: dict[str, str] = {
    "USD": "DFF",
    "EUR": "ECBDFR",
    "GBP": "IUDSOIA",
    "JPY": "INTDSRJPM193N",
    "CHF": "IRSTCI01CHM156N",
    "CAD": "IRSTCI01CAM156N",
    "AUD": "IRSTCI01AUM156N",
    "NZD": "IRSTCI01NZM156N",
}

G10_CURRENCIES = tuple(POLICY_RATE_SERIES.keys())

# Policy rates in annual decimal never exceed 100% for G10; |level| > 1 ⇒ percent.
_PERCENT_LEVEL_ABS_MAX = 1.0


def _normalize_ccy(ccy: str) -> str:
    return str(ccy).strip().upper()


def _normalize_pair(pair: str) -> tuple[str, str]:
    p = str(pair).strip().upper().replace("/", "").replace("_", "").replace("-", "")
    if len(p) != 6:
        raise ValueError(f"expected 6-letter FX pair like EURUSD, got {pair!r}")
    return p[:3], p[3:]


def policy_rates_as_annual_decimal(
    rates: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Convert FRED-style percent levels to annual decimals when needed.

    Idempotent: if no finite ``|level| > 1``, values already look like decimals
    and are returned unchanged (copy).
    """
    if rates is None or (hasattr(rates, "empty") and rates.empty):
        return rates
    out = rates.copy()
    if isinstance(out, pd.DataFrame):
        flat = pd.to_numeric(out.to_numpy().ravel(), errors="coerce")
    else:
        flat = pd.to_numeric(out, errors="coerce").to_numpy(dtype=float)
    finite = np.asarray(flat, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return out
    if float(np.nanmax(np.abs(finite))) <= _PERCENT_LEVEL_ABS_MAX:
        return out
    return out / 100.0


def ensure_policy_rates_frame(
    rates_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Normalize a policy-rate panel for swap / carry consumers.

    - Accepts a DatetimeIndex **or** a ``date`` column (research parquet layout).
    - Converts percent → annual decimal via :func:`policy_rates_as_annual_decimal`.
    - Returns a sorted copy with ``date`` as the index name; empty/None → empty.
    """
    if rates_df is None or rates_df.empty:
        return pd.DataFrame()
    out = rates_df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.index.name = "date"
    num_cols = list(out.select_dtypes(include=["number"]).columns)
    if num_cols:
        converted = policy_rates_as_annual_decimal(out[num_cols])
        out[num_cols] = converted
    return out


def fetch_policy_rate(
    ccy: str,
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
) -> pd.Series:
    """Fetch one G10 policy rate as an **annual decimal** (FRED percent / 100)."""
    currency = _normalize_ccy(ccy)
    series_id = POLICY_RATE_SERIES.get(currency)
    if series_id is None:
        raise KeyError(
            f"no POLICY_RATE_SERIES entry for {currency!r}; "
            f"known: {sorted(POLICY_RATE_SERIES)}"
        )
    s = fetch_fred_series(series_id, start=start, end=end)
    s = policy_rates_as_annual_decimal(s)
    s.name = currency
    logger.debug("policy rate %s ← FRED %s (%d obs)", currency, series_id, len(s))
    return s


def fetch_all_g10_policy_rates(
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Wide DataFrame of G10 policy rates as **annual decimals** (columns = ISO).

    Forward-fills across weekends / missing business days after an outer join
    onto a daily calendar spanning the observed range.
    """
    series_list: list[pd.Series] = []
    for ccy in G10_CURRENCIES:
        try:
            series_list.append(fetch_policy_rate(ccy, start=start, end=end))
        except Exception as exc:
            logger.warning("skipping policy rate for %s: %s", ccy, exc)

    if not series_list:
        return pd.DataFrame()

    wide = pd.concat(series_list, axis=1).sort_index()
    # Daily calendar + weekend forward-fill
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full_idx).ffill()
    wide.index.name = "date"
    return ensure_policy_rates_frame(wide)


def rate_differential(pair: str, rates_df: pd.DataFrame) -> pd.Series:
    """
    Base − quote policy rate for an FX pair (e.g. EURUSD → EUR − USD).

    ``rates_df`` may be percent or decimal and may use a ``date`` column; it is
    normalized via :func:`ensure_policy_rates_frame` first.
    """
    rates = ensure_policy_rates_frame(rates_df)
    base, quote = _normalize_pair(pair)
    if base not in rates.columns:
        raise KeyError(f"base currency {base} missing from rates_df columns")
    if quote not in rates.columns:
        raise KeyError(f"quote currency {quote} missing from rates_df columns")
    diff = rates[base].astype(float) - rates[quote].astype(float)
    diff.name = f"{base}{quote}_rate_diff"
    return diff
