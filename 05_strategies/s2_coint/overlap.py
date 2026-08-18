"""H-007 / H-009 conflict helpers: shared tickers and score × confidence."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def pair_tickers(pair_id: str, ticker_y: str, ticker_x: str) -> frozenset[str]:
    return frozenset({str(ticker_y), str(ticker_x)})


def shares_leg(a: Iterable[str], b: Iterable[str]) -> bool:
    return not frozenset(a).isdisjoint(frozenset(b))


def score_confidence(score: float, adf_pvalue: float) -> float:
    """Tie-break / priority: ``|score| * (1 - p_ADF)``. NaN → 0."""
    import math

    if not math.isfinite(score):
        return 0.0
    p = adf_pvalue if math.isfinite(adf_pvalue) else 1.0
    p = min(max(p, 0.0), 1.0)
    return abs(float(score)) * (1.0 - p)


def pick_same_bar_winner(
    candidates: Sequence[tuple[str, float]],
) -> str | None:
    """``candidates`` is ``(pair_id, score_x_conf)``; highest wins. None if empty."""
    if not candidates:
        return None
    best = max(candidates, key=lambda t: (t[1], t[0]))
    return best[0]


def corr_blocks_candidate(
    pair_id: str,
    score_conf: float,
    *,
    open_ids: Iterable[str],
    same_bar_candidates: Sequence[tuple[str, float]],
    abs_rho: dict[tuple[str, str], float],
    k: float,
) -> bool:
    """H-009: block a new pair when |ρ̂| > k vs an open pair or a better same-bar rival."""
    for oid in open_ids:
        key = tuple(sorted((pair_id, oid)))
        rho = abs_rho.get(key)
        if rho is not None and abs(float(rho)) > float(k):
            return True
    for oid, osc in same_bar_candidates:
        if oid == pair_id:
            continue
        key = tuple(sorted((pair_id, oid)))
        rho = abs_rho.get(key)
        if rho is None:
            continue
        if abs(float(rho)) > float(k) and (osc, oid) > (score_conf, pair_id):
            return True
    return False
