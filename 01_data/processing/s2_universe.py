"""S2 universe helpers: nested pool loading and pair candidacy (not cointegration math)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

from data.processing.s2_universe_pools import (
    RESEARCH_IS_END_BY_UNIVERSE,
    S2_POOLS,
    SHELVED_UNIVERSES,
)

__all__ = [
    "RESEARCH_IS_END_BY_UNIVERSE",
    "SHELVED_UNIVERSES",
    "iter_leaf_pools",
    "iter_pool_pairs",
    "iter_same_venue_pairs",
    "load_s2_pools",
    "pool_of_pair",
    "pool_path_for_pair",
    "pool_tickers",
    "research_is_end",
    "ticker_venue_key",
]

# US share-class suffixes: canonical ``BF.B`` is a class line, not an exchange.
_US_CLASS_SUFFIXES: frozenset[str] = frozenset({"A", "B", "C", "D", "K"})


def load_s2_pools(label: str | None = None) -> dict | list:
    """Nested pools for one universe letter, or every universe when ``label`` is None.

    Replaces the old three-line ``s2_universes.csv`` loader: the number of universes is
    dynamic and pools carry arbitrary nesting depth.
    """
    if label is None:
        return dict(S2_POOLS)
    key = str(label).strip().upper()
    if key not in S2_POOLS:
        raise KeyError(
            f"unknown universe {label!r}; available: {sorted(S2_POOLS)}"
        )
    value = S2_POOLS[key]
    return dict(value) if isinstance(value, Mapping) else list(value)


def research_is_end(label: str) -> str:
    """Pre-registered research-IS end for a universe letter."""
    key = str(label).strip().upper()
    if key not in RESEARCH_IS_END_BY_UNIVERSE:
        raise KeyError(f"no RESEARCH_IS_END registered for universe {label!r}")
    return RESEARCH_IS_END_BY_UNIVERSE[key]


def _is_leaf(node) -> bool:
    """Leaf pool = a sequence of ticker strings (not a mapping)."""
    if isinstance(node, Mapping) or isinstance(node, str):
        return False
    if not isinstance(node, Sequence):
        return False
    return all(isinstance(item, str) for item in node)


def iter_leaf_pools(nested, *, _path: tuple[str, ...] = ()) -> list[tuple[str, list[str]]]:
    """Recursively collect ``(pool_name, tickers)`` for every leaf list at any depth.

    ``pool_name`` is the dotted path of labels down to the leaf (e.g. ``F.MC.es_banks``),
    so pools stay identifiable no matter how the tree is re-nested.
    """
    if _is_leaf(nested):
        name = ".".join(_path) if _path else "root"
        return [(name, [str(t) for t in nested])]

    out: list[tuple[str, list[str]]] = []
    if isinstance(nested, Mapping):
        for label, child in nested.items():
            out.extend(iter_leaf_pools(child, _path=(*_path, str(label))))
        return out
    if isinstance(nested, Sequence) and not isinstance(nested, str):
        # Unlabelled nesting (list of lists) - index the level so names stay unique.
        for idx, child in enumerate(nested):
            out.extend(iter_leaf_pools(child, _path=(*_path, str(idx))))
        return out
    raise TypeError(f"unsupported pool node type: {type(nested)!r}")


def pool_tickers(nested) -> list[str]:
    """Every ticker under ``nested``, de-duplicated, first-seen order preserved."""
    seen: list[str] = []
    for _, tickers in iter_leaf_pools(nested):
        for t in tickers:
            if t not in seen:
                seen.append(t)
    return seen


def iter_pool_pairs(nested) -> list[tuple[str, str]]:
    """Candidate pairs formed **only within** each leaf pool, at any nesting depth.

    Never pairs across pools and never pairs a ticker with itself. Legs are sorted for a
    stable pre-Engle-Granger ordering (EG later re-orients ``y|x`` by the lower p-value).
    Same-venue is asserted, so a leaf pool that accidentally mixes exchanges raises rather
    than silently producing an unalignable pair.
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for name, tickers in iter_leaf_pools(nested):
        unique = list(dict.fromkeys(tickers))
        for a, b in combinations(unique, 2):
            if ticker_venue_key(a) != ticker_venue_key(b):
                raise ValueError(
                    f"pool {name!r} mixes venues: {a} ({ticker_venue_key(a)}) "
                    f"vs {b} ({ticker_venue_key(b)})"
                )
            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue
            seen.add(pair)
            out.append((pair[0], pair[1]))
    return out


def pool_of_pair(nested) -> dict[str, str]:
    """Map ``"a|b"`` (both orientations) to its leaf pool name, for per-pool caps."""
    out: dict[str, str] = {}
    for name, tickers in iter_leaf_pools(nested):
        unique = list(dict.fromkeys(tickers))
        for a, b in combinations(unique, 2):
            out[f"{a}|{b}"] = name
            out[f"{b}|{a}"] = name
    return out


def pool_path_for_pair(pair_id: str, nested) -> str:
    """Leaf pool name for a ``ticker_y|ticker_x`` id, orientation-insensitive."""
    lookup = pool_of_pair(nested)
    key = str(pair_id)
    if key in lookup:
        return lookup[key]
    parts = key.split("|")
    if len(parts) == 2:
        flipped = f"{parts[1]}|{parts[0]}"
        if flipped in lookup:
            return lookup[flipped]
    raise KeyError(f"pair {pair_id!r} is not inside any leaf pool")


def ticker_venue_key(ticker: str) -> str:
    """Same-venue key for candidate pairs (blocks e.g. HK/JP or Madrid/Milan crosses).

    Rules (checked in order):
    - Yahoo FX suffix ``=X`` -> ``FX``
    - Yahoo crypto suffix ``-USD`` -> ``CRYPTO``
    - US share-class line (``BF.B``, ``HEI.A``, ``CWEN.A``) -> ``US``; the dotted segment is
      a class letter, not an exchange, so both classes must share a key to be pairable.
    - exchange suffix after the last ``.`` (``.HK`` -> ``HK``, ``.MC`` -> ``MC``)
    - plain alphabetic symbol (``GOOGL``, ``AMT``) -> ``US``
    - else the full ticker (unknown symbols never silently cross-match)
    """
    t = ticker.strip()
    if not t:
        raise ValueError("ticker must be a non-empty string")

    if t.endswith("=X"):
        return "FX"
    if t.upper().endswith("-USD"):
        return "CRYPTO"

    if "." in t:
        stem, _, suffix = t.rpartition(".")
        suffix_up = suffix.upper()
        # A single-letter suffix on an alphabetic stem is a US share class, not a venue.
        if stem.isalpha() and suffix_up in _US_CLASS_SUFFIXES:
            return "US"
        return suffix_up

    if t.isalpha():
        return "US"
    return t


def iter_same_venue_pairs(tickers: Sequence[str]) -> list[tuple[str, str]]:
    """Unordered same-venue pairs across a flat ticker list; legs sorted.

    Retained for flat universes and sanity checks. Pool-scoped universes should use
    ``iter_pool_pairs`` so candidates never cross a sector boundary.
    """
    out: list[tuple[str, str]] = []
    for a, b in combinations(tickers, 2):
        if ticker_venue_key(a) != ticker_venue_key(b):
            continue
        left, right = sorted((a, b))
        out.append((left, right))
    return out
