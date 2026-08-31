"""Challenge-fee economics (parameterized; not official FTMO prices)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pay_once_cashflows(
    passed_both: pd.Series | np.ndarray,
    funded_surplus: pd.Series | np.ndarray,
    *,
    fee: float,
    profit_split: float,
    verification_fee: float = 0.0,
) -> pd.Series:
    """One purchase: ``-fees + 1{both} * split * max(surplus, 0)``."""
    passed = np.asarray(passed_both, dtype=bool)
    surplus = np.asarray(funded_surplus, dtype=float)
    payout = np.where(passed, float(profit_split) * np.maximum(surplus, 0.0), 0.0)
    cash = -float(fee) - float(verification_fee) + payout
    index = None
    if isinstance(passed_both, pd.Series):
        index = passed_both.index
    elif isinstance(funded_surplus, pd.Series):
        index = funded_surplus.index
    return pd.Series(cash, index=index, name="cashflow")


def economic_ev_summary(
    cashflows: pd.Series,
    *,
    level: float = 0.95,
    pass_rate_both: float | None = None,
    fee: float | None = None,
) -> pd.Series:
    """Pathwise EV, percentile CI, ``P(cashflow <= 0)``, do-not-take flag.

    ``do_not_take`` if mean EV ≤ 0 or the lower CI bound ≤ 0 (fee not clearly earned).
    Does not reuse HAC-on-returns t-stats from the EV vs-SPY package.
    """
    x = pd.to_numeric(cashflows, errors="coerce").astype(float).dropna()
    if x.empty:
        return pd.Series(
            {
                "ev": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "p_ev_le_0": float("nan"),
                "do_not_take": True,
                "n_paths": 0,
                "ev_per_day": float("nan"),
                "expected_retries": float("nan"),
                "ev_retry_until_pass": float("nan"),
            },
            name="economics",
        )
    alpha = (1.0 - float(level)) / 2.0
    ev = float(x.mean())
    ci_low = float(x.quantile(alpha))
    ci_high = float(x.quantile(1.0 - alpha))
    p_le0 = float((x.to_numpy() <= 0.0).mean())
    do_not = bool(ev <= 0.0 or ci_low <= 0.0)
    p = float(pass_rate_both) if pass_rate_both is not None else float("nan")
    retries = float((1.0 - p) / p) if np.isfinite(p) and p > 0.0 else float("nan")
    payout = x + (float(fee) if fee is not None else 0.0)
    if np.isfinite(p) and p > 0.0 and fee is not None:
        # Conditional mean payout among paths, using all cashflows+fee as payout
        # including zeros on fails: E[payout]/p - fee/p = E[payout|pass] - fee/p
        ev_retry = float(payout.mean() / p - float(fee) / p)
    else:
        ev_retry = float("nan")
    return pd.Series(
        {
            "ev": ev,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_ev_le_0": p_le0,
            "do_not_take": do_not,
            "n_paths": float(x.shape[0]),
            "expected_retries": retries,
            "ev_retry_until_pass": ev_retry,
            "level": float(level),
        },
        name="economics",
    )


def attach_economics(
    two_step: pd.DataFrame,
    *,
    fee: float,
    profit_split: float,
    verification_fee: float = 0.0,
    horizon_days: float | None = None,
    level: float = 0.95,
) -> tuple[pd.Series, pd.Series]:
    """Cashflows per path plus EV summary (optional EV/day using ``horizon_days``)."""
    cash = pay_once_cashflows(
        two_step["passed_both"],
        two_step["funded_surplus"],
        fee=fee,
        profit_split=profit_split,
        verification_fee=verification_fee,
    )
    p_both = float(two_step["passed_both"].mean()) if len(two_step) else float("nan")
    summary = economic_ev_summary(
        cash, level=level, pass_rate_both=p_both, fee=fee + verification_fee
    )
    if horizon_days is not None and np.isfinite(horizon_days) and horizon_days > 0:
        summary["ev_per_day"] = float(summary["ev"]) / float(horizon_days)
    else:
        summary["ev_per_day"] = float("nan")
    return cash, summary


def geometric_attempts_until_pass(
    p_pass: float,
    n_draw: int,
    *,
    random_seed: int = 0,
    max_attempts: int = 500,
) -> np.ndarray:
    """I.i.d. retries until first two-step pass (histogram helper)."""
    p = float(p_pass)
    rng = np.random.default_rng(random_seed)
    if not np.isfinite(p) or p <= 0.0:
        return np.full(int(n_draw), float(max_attempts), dtype=float)
    if p >= 1.0:
        return np.ones(int(n_draw), dtype=float)
    # Geometric number of trials until first success, support starting at 1
    draws = rng.geometric(p, size=int(n_draw)).astype(float)
    return np.minimum(draws, float(max_attempts))
