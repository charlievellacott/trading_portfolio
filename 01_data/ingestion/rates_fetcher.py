"""
G10 policy / short-rate series via FRED for S3 FX carry / rate differentials.

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


def _normalize_ccy(ccy: str) -> str:
    return str(ccy).strip().upper()


def _normalize_pair(pair: str) -> tuple[str, str]:
    p = str(pair).strip().upper().replace("/", "").replace("_", "").replace("-", "")
    if len(p) != 6:
        raise ValueError(f"expected 6-letter FX pair like EURUSD, got {pair!r}")
    return p[:3], p[3:]


def fetch_policy_rate(
    ccy: str,
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
) -> pd.Series:
    """Fetch the mapped policy / short-rate series for one G10 currency."""
    currency = _normalize_ccy(ccy)
    series_id = POLICY_RATE_SERIES.get(currency)
    if series_id is None:
        raise KeyError(
            f"no POLICY_RATE_SERIES entry for {currency!r}; "
            f"known: {sorted(POLICY_RATE_SERIES)}"
        )
    s = fetch_fred_series(series_id, start=start, end=end)
    s.name = currency
    logger.debug("policy rate %s ← FRED %s (%d obs)", currency, series_id, len(s))
    return s


def fetch_all_g10_policy_rates(
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Wide DataFrame of G10 policy rates (columns = currency codes).

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
    return wide


def rate_differential(pair: str, rates_df: pd.DataFrame) -> pd.Series:
    """
    Base − quote policy rate for an FX pair (e.g. EURUSD → EUR − USD).

    ``rates_df`` must have currency columns (as from
    ``fetch_all_g10_policy_rates``).
    """
    base, quote = _normalize_pair(pair)
    if base not in rates_df.columns:
        raise KeyError(f"base currency {base} missing from rates_df columns")
    if quote not in rates_df.columns:
        raise KeyError(f"quote currency {quote} missing from rates_df columns")
    diff = rates_df[base].astype(float) - rates_df[quote].astype(float)
    diff.name = f"{base}{quote}_rate_diff"
    return diff
