"""Live performance ledger: signals, fills, positions, and equity by sleeve."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import pandas as pd

from data.repo_paths import repo_root

# Constants
STRATEGY_S1_EQUITIES = "s1_equities"
BROKER_ALPACA = "alpaca"
DEFAULT_SLEEVE_WEIGHTS: dict[str, float] = {STRATEGY_S1_EQUITIES: 1.0}

_CACHE_REL = os.path.join("09_performance", "cache", "live")
_META_NAME = "meta.json"
_SIGNALS_NAME = "signals.parquet"
_FILLS_NAME = "fills.parquet"
_POSITIONS_NAME = "positions.parquet"
_EQUITY_NAME = "equity_daily.parquet"

_SIGNAL_KEYS = ("decision_date", "strategy_id", "ticker")
_FILL_KEYS = ("order_id",)
_POSITION_KEYS = ("ts", "strategy_id", "ticker")
_EQUITY_KEYS = ("date", "level", "strategy_id")

_SIGNAL_COLS = [
    "ts",
    "decision_date",
    "strategy_id",
    "broker_id",
    "ticker",
    "score",
    "weight",
    "feature_date",
    "run_id",
]
_FILL_COLS = [
    "ts",
    "strategy_id",
    "broker_id",
    "ticker",
    "side",
    "qty",
    "price",
    "order_id",
    "fill_type",
    "run_id",
]
_POSITION_COLS = [
    "ts",
    "strategy_id",
    "broker_id",
    "ticker",
    "qty",
    "avg_entry",
    "market_value",
    "unrealized_pl",
]
_EQUITY_COLS = [
    "date",
    "level",
    "strategy_id",
    "equity",
    "allocated_equity",
    "cash",
]


# Subroutines
def _cache_dir(cache_dir: str | None = None) -> str:
    if cache_dir is not None:
        return os.path.abspath(cache_dir)
    return os.path.join(repo_root(), _CACHE_REL)


def _ensure_cache_dir(cache_dir: str | None = None) -> str:
    path = _cache_dir(cache_dir)
    os.makedirs(path, exist_ok=True)
    return path


def _table_path(name: str, cache_dir: str | None = None) -> str:
    return os.path.join(_ensure_cache_dir(cache_dir), name)


def _meta_path(cache_dir: str | None = None) -> str:
    return _table_path(_META_NAME, cache_dir)


def _default_meta() -> dict[str, Any]:
    return {
        "portfolio_go_live": None,
        "sleeve_weights": dict(DEFAULT_SLEEVE_WEIGHTS),
        "updated_at": None,
    }


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {
        str(k): float(v)
        for k, v in dict(weights).items()
        if float(v) > 0.0
    }
    if not cleaned:
        raise ValueError("sleeve_weights must contain at least one positive weight")
    total = sum(cleaned.values())
    return {k: v / total for k, v in cleaned.items()}


def _now_ts() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _as_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _as_date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(_as_timestamp(value).date())


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_table(name: str, columns: list[str], cache_dir: str | None = None) -> pd.DataFrame:
    path = _table_path(name, cache_dir)
    if not os.path.isfile(path):
        return _empty_frame(columns)
    frame = pd.read_parquet(path)
    for col in columns:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame.loc[:, columns].copy()


def _key_series(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
    parts = []
    for key in keys:
        col = frame[key]
        if key in {"decision_date", "date", "ts"}:
            col = pd.to_datetime(col, errors="coerce").astype(str)
        else:
            col = col.fillna("").astype(str)
        parts.append(col)
    out = parts[0]
    for part in parts[1:]:
        out = out + "|" + part
    return out


def _upsert_parquet(
    name: str,
    new_rows: pd.DataFrame,
    *,
    columns: list[str],
    keys: tuple[str, ...],
    cache_dir: str | None = None,
) -> pd.DataFrame:
    if new_rows is None or new_rows.empty:
        return _read_table(name, columns, cache_dir)

    incoming = new_rows.copy()
    for col in columns:
        if col not in incoming.columns:
            incoming[col] = pd.NA
    incoming = incoming.loc[:, columns]

    existing = _read_table(name, columns, cache_dir)
    if existing.empty:
        combined = incoming
    else:
        incoming_keys = set(_key_series(incoming, keys))
        keep_mask = ~_key_series(existing, keys).isin(incoming_keys)
        combined = pd.concat(
            [existing.loc[keep_mask], incoming],
            ignore_index=True,
        )
    combined = combined.loc[~_key_series(combined, keys).duplicated(keep="last")]

    path = _table_path(name, cache_dir)
    combined.to_parquet(path, index=False)
    return combined.reset_index(drop=True)


def _write_meta(meta: dict[str, Any], cache_dir: str | None = None) -> dict[str, Any]:
    path = _meta_path(cache_dir)
    _ensure_cache_dir(cache_dir)
    payload = dict(meta)
    payload["updated_at"] = str(_now_ts())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def _signed_qty_from_fill(side: str, qty: float) -> float:
    s = str(side).strip().lower()
    q = abs(float(qty))
    if s in {"buy", "long", "cover"}:
        return q
    if s in {"sell", "short"}:
        return -q
    raise ValueError(f"unknown fill side: {side!r}")


def _positions_from_fills(
    fills: pd.DataFrame,
    *,
    strategy_id: str | None = None,
) -> dict[tuple[str, str], float]:
    if fills is None or fills.empty:
        return {}
    frame = fills.copy()
    if strategy_id is not None:
        frame = frame.loc[frame["strategy_id"].astype(str) == str(strategy_id)]
    if frame.empty:
        return {}
    frame = frame.sort_values("ts")
    qty_map: dict[tuple[str, str], float] = {}
    for _, row in frame.iterrows():
        qty = abs(float(row.get("qty", 0.0) or 0.0))
        if qty < 1e-12:
            continue
        sid = str(row["strategy_id"])
        ticker = str(row["ticker"]).strip().upper()
        delta = _signed_qty_from_fill(row["side"], qty)
        key = (sid, ticker)
        qty_map[key] = qty_map.get(key, 0.0) + delta
        if abs(qty_map[key]) < 1e-10:
            del qty_map[key]
    return qty_map


def _apply_go_live(series: pd.Series, go_live: Any) -> pd.Series:
    if series is None or series.empty:
        return series
    if go_live is None or (isinstance(go_live, float) and pd.isna(go_live)):
        return series
    start = _as_date(go_live)
    idx = pd.DatetimeIndex(pd.to_datetime(series.index))
    return series.loc[idx >= start]


def _equity_frame_to_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    series = pd.Series(
        out["equity"].astype(float).values,
        index=pd.DatetimeIndex(out["date"]),
        name="equity",
    )
    series = series[~series.index.duplicated(keep="last")]
    return series.sort_index()


# Main
def get_meta(cache_dir: str | None = None) -> dict[str, Any]:
    """Load ledger meta; create defaults when missing."""
    path = _meta_path(cache_dir)
    if not os.path.isfile(path):
        return _write_meta(_default_meta(), cache_dir)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    meta = _default_meta()
    meta.update(raw or {})
    meta["sleeve_weights"] = _normalize_weights(
        meta.get("sleeve_weights") or DEFAULT_SLEEVE_WEIGHTS
    )
    return meta


def set_portfolio_go_live(
    go_live: Any,
    *,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Set shared portfolio go-live date (``None`` keeps full history for display)."""
    meta = get_meta(cache_dir)
    if go_live is None:
        meta["portfolio_go_live"] = None
    else:
        meta["portfolio_go_live"] = str(_as_date(go_live).date())
    return _write_meta(meta, cache_dir)


def set_sleeve_weights(
    weights: dict[str, float],
    *,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Replace sleeve capital weights (renormalized to sum to 1)."""
    meta = get_meta(cache_dir)
    meta["sleeve_weights"] = _normalize_weights(weights)
    return _write_meta(meta, cache_dir)


def log_signals(
    strategy_id: str,
    weights_df: pd.DataFrame,
    *,
    broker_id: str = BROKER_ALPACA,
    run_id: str | None = None,
    ts: Any | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Upsert signal / weight rows for a decision date cross-section."""
    if weights_df is None or weights_df.empty:
        return load_signals(cache_dir=cache_dir)

    frame = weights_df.copy()
    if "ticker" not in frame.columns or "weight" not in frame.columns:
        raise ValueError("weights_df must include ticker and weight columns")

    run = run_id or str(uuid.uuid4())
    stamp = _as_timestamp(ts) if ts is not None else _now_ts()

    if "date" in frame.columns:
        decision = pd.to_datetime(frame["date"])
    elif "decision_date" in frame.columns:
        decision = pd.to_datetime(frame["decision_date"])
    else:
        decision = pd.Series([stamp.normalize()] * len(frame), index=frame.index)

    score = frame["score"] if "score" in frame.columns else pd.NA
    feature_date = (
        pd.to_datetime(frame["feature_date"])
        if "feature_date" in frame.columns
        else pd.NaT
    )

    rows = pd.DataFrame(
        {
            "ts": stamp,
            "decision_date": pd.to_datetime(decision).dt.normalize(),
            "strategy_id": str(strategy_id),
            "broker_id": str(broker_id),
            "ticker": frame["ticker"].astype(str).str.strip().str.upper(),
            "score": score,
            "weight": frame["weight"].astype(float),
            "feature_date": feature_date,
            "run_id": run,
        }
    )
    return _upsert_parquet(
        _SIGNALS_NAME,
        rows,
        columns=_SIGNAL_COLS,
        keys=_SIGNAL_KEYS,
        cache_dir=cache_dir,
    )


def log_fills(
    fills: list[dict[str, Any]] | pd.DataFrame,
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Upsert fill rows keyed by ``order_id``."""
    if fills is None or (isinstance(fills, pd.DataFrame) and fills.empty):
        return load_fills(cache_dir=cache_dir)
    if isinstance(fills, list) and not fills:
        return load_fills(cache_dir=cache_dir)

    frame = pd.DataFrame(fills) if not isinstance(fills, pd.DataFrame) else fills.copy()
    required = {"strategy_id", "ticker", "side", "qty", "order_id", "fill_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"log_fills missing columns: {sorted(missing)}")

    if "ts" not in frame.columns:
        frame["ts"] = _now_ts()
    if "broker_id" not in frame.columns:
        frame["broker_id"] = BROKER_ALPACA
    if "price" not in frame.columns:
        frame["price"] = pd.NA
    if "run_id" not in frame.columns:
        frame["run_id"] = pd.NA

    frame["ts"] = pd.to_datetime(frame["ts"])
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["broker_id"] = frame["broker_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["side"] = frame["side"].astype(str).str.strip().str.lower()
    frame["qty"] = frame["qty"].astype(float)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["order_id"] = frame["order_id"].astype(str)
    frame["fill_type"] = frame["fill_type"].astype(str)
    frame["run_id"] = frame["run_id"].astype(str)

    return _upsert_parquet(
        _FILLS_NAME,
        frame,
        columns=_FILL_COLS,
        keys=_FILL_KEYS,
        cache_dir=cache_dir,
    )


def log_session_fills(
    *,
    strategy_id: str,
    broker_id: str = BROKER_ALPACA,
    run_id: str | None = None,
    liquidate: list[dict[str, Any]] | None = None,
    entries: list[dict[str, Any]] | None = None,
    uncovered: list[dict[str, Any]] | None = None,
    stops: list[dict[str, Any]] | None = None,
    ts: Any | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Convert runner session dicts into ledger fill rows and upsert."""
    stamp = _as_timestamp(ts) if ts is not None else _now_ts()
    run = run_id or str(uuid.uuid4())
    rows: list[dict[str, Any]] = []

    for i, row in enumerate(liquidate or []):
        ticker = str(row.get("ticker", "")).strip().upper()
        qty = abs(float(row.get("qty", 0.0) or 0.0))
        if not ticker or qty <= 0:
            continue
        order_id = str(row.get("order_id") or f"{run}:liquidate:{ticker}:{i}")
        rows.append(
            {
                "ts": stamp,
                "strategy_id": strategy_id,
                "broker_id": broker_id,
                "ticker": ticker,
                "side": "sell" if float(row.get("qty", 0.0) or 0.0) > 0 else "buy",
                "qty": qty,
                "price": row.get("price", pd.NA),
                "order_id": order_id,
                "fill_type": "liquidate",
                "run_id": run,
            }
        )

    for row in entries or []:
        ticker = str(row.get("ticker", "")).strip().upper()
        qty = abs(float(row.get("fill_qty", row.get("qty", 0.0)) or 0.0))
        if not ticker or qty < 1:
            continue
        is_long = bool(row.get("is_long", True))
        rows.append(
            {
                "ts": stamp,
                "strategy_id": strategy_id,
                "broker_id": broker_id,
                "ticker": ticker,
                "side": "buy" if is_long else "sell",
                "qty": qty,
                "price": row.get("fill_px", row.get("price", pd.NA)),
                "order_id": str(row["order_id"]),
                "fill_type": "entry",
                "run_id": run,
            }
        )

    for i, row in enumerate(uncovered or []):
        if not row.get("closed"):
            continue
        ticker = str(row.get("ticker", "")).strip().upper()
        qty = abs(float(row.get("qty", 0.0) or 0.0))
        if not ticker or qty <= 0:
            continue
        order_id = str(row.get("order_id") or f"{run}:uncovered:{ticker}:{i}")
        rows.append(
            {
                "ts": stamp,
                "strategy_id": strategy_id,
                "broker_id": broker_id,
                "ticker": ticker,
                "side": "sell",
                "qty": qty,
                "price": row.get("price", pd.NA),
                "order_id": order_id,
                "fill_type": "uncovered_close",
                "run_id": run,
            }
        )

    for row in stops or []:
        # Resting stop submissions are tracked by order_id for later sync_stop_fills.
        # Only persist when already filled; otherwise keep order_id on a zero-qty placeholder.
        ticker = str(row.get("ticker", "")).strip().upper()
        order_id = row.get("order_id")
        if not ticker or order_id is None:
            continue
        fill_qty = abs(float(row.get("fill_qty", row.get("qty", 0.0)) or 0.0))
        if fill_qty <= 0:
            continue
        rows.append(
            {
                "ts": stamp,
                "strategy_id": strategy_id,
                "broker_id": broker_id,
                "ticker": ticker,
                "side": str(row.get("side", "sell")).lower(),
                "qty": fill_qty,
                "price": row.get("fill_px", row.get("price", pd.NA)),
                "order_id": str(order_id),
                "fill_type": "stop",
                "run_id": run,
            }
        )

    return log_fills(rows, cache_dir=cache_dir)


def log_positions_snapshot(
    positions: list[dict[str, Any]] | pd.DataFrame,
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Upsert position snapshot rows."""
    if positions is None or (isinstance(positions, pd.DataFrame) and positions.empty):
        return load_positions(cache_dir=cache_dir)
    if isinstance(positions, list) and not positions:
        return load_positions(cache_dir=cache_dir)

    frame = (
        pd.DataFrame(positions)
        if not isinstance(positions, pd.DataFrame)
        else positions.copy()
    )
    required = {"strategy_id", "ticker", "qty"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"log_positions_snapshot missing columns: {sorted(missing)}")

    if "ts" not in frame.columns:
        frame["ts"] = _now_ts()
    if "broker_id" not in frame.columns:
        frame["broker_id"] = BROKER_ALPACA
    for col in ("avg_entry", "market_value", "unrealized_pl"):
        if col not in frame.columns:
            frame[col] = pd.NA

    frame["ts"] = pd.to_datetime(frame["ts"])
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["broker_id"] = frame["broker_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["qty"] = frame["qty"].astype(float)

    return _upsert_parquet(
        _POSITIONS_NAME,
        frame,
        columns=_POSITION_COLS,
        keys=_POSITION_KEYS,
        cache_dir=cache_dir,
    )


def log_equity_daily(
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Upsert daily equity rows for portfolio and/or strategy levels."""
    if rows is None or (isinstance(rows, pd.DataFrame) and rows.empty):
        return load_equity_daily(cache_dir=cache_dir)
    if isinstance(rows, list) and not rows:
        return load_equity_daily(cache_dir=cache_dir)

    frame = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    required = {"date", "level", "equity"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"log_equity_daily missing columns: {sorted(missing)}")

    if "strategy_id" not in frame.columns:
        frame["strategy_id"] = pd.NA
    if "allocated_equity" not in frame.columns:
        frame["allocated_equity"] = pd.NA
    if "cash" not in frame.columns:
        frame["cash"] = pd.NA

    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["level"] = frame["level"].astype(str)
    frame["strategy_id"] = frame["strategy_id"].where(
        frame["strategy_id"].notna(), other=""
    )
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame.loc[frame["level"] == "portfolio", "strategy_id"] = ""
    frame["equity"] = frame["equity"].astype(float)

    return _upsert_parquet(
        _EQUITY_NAME,
        frame,
        columns=_EQUITY_COLS,
        keys=_EQUITY_KEYS,
        cache_dir=cache_dir,
    )


def load_signals(cache_dir: str | None = None) -> pd.DataFrame:
    return _read_table(_SIGNALS_NAME, _SIGNAL_COLS, cache_dir)


def load_fills(cache_dir: str | None = None) -> pd.DataFrame:
    return _read_table(_FILLS_NAME, _FILL_COLS, cache_dir)


def load_positions(cache_dir: str | None = None) -> pd.DataFrame:
    return _read_table(_POSITIONS_NAME, _POSITION_COLS, cache_dir)


def load_equity_daily(cache_dir: str | None = None) -> pd.DataFrame:
    return _read_table(_EQUITY_NAME, _EQUITY_COLS, cache_dir)


def portfolio_equity_series(cache_dir: str | None = None) -> pd.Series:
    """Portfolio equity curve; truncated when ``portfolio_go_live`` is set."""
    meta = get_meta(cache_dir)
    frame = load_equity_daily(cache_dir)
    if frame.empty:
        return pd.Series(dtype=float, name="equity")
    port = frame.loc[frame["level"].astype(str) == "portfolio"]
    series = _equity_frame_to_series(port)
    return _apply_go_live(series, meta.get("portfolio_go_live"))


def strategy_equity_series(
    strategy_id: str,
    *,
    cache_dir: str | None = None,
) -> pd.Series:
    """Strategy sleeve equity curve; truncated when ``portfolio_go_live`` is set."""
    meta = get_meta(cache_dir)
    frame = load_equity_daily(cache_dir)
    if frame.empty:
        return pd.Series(dtype=float, name="equity")
    sleeve = frame.loc[
        (frame["level"].astype(str) == "strategy")
        & (frame["strategy_id"].astype(str) == str(strategy_id))
    ]
    series = _equity_frame_to_series(sleeve)
    return _apply_go_live(series, meta.get("portfolio_go_live"))


def sleeve_qty_map(
    *,
    strategy_id: str | None = None,
    cache_dir: str | None = None,
) -> dict[tuple[str, str], float]:
    """Rebuild signed open qty per (strategy_id, ticker) from the fill ledger."""
    return _positions_from_fills(load_fills(cache_dir), strategy_id=strategy_id)


def sync_stop_fills(
    broker: Any,
    strategy_id: str,
    *,
    since: Any | None = None,
    broker_id: str = BROKER_ALPACA,
    run_id: str | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """
    Pull newly filled stop orders from the broker and append to the fill ledger.

    ``broker`` must implement ``get_filled_orders(after=..., until=...)``.
    """
    fills = load_fills(cache_dir)
    if since is None:
        stop_like = fills.loc[fills["fill_type"].astype(str) == "stop"] if not fills.empty else fills
        if stop_like is not None and not stop_like.empty:
            since = pd.to_datetime(stop_like["ts"]).max()
        elif not fills.empty:
            since = pd.to_datetime(fills["ts"]).max()

    after = _as_timestamp(since) if since is not None else None
    raw = list(broker.get_filled_orders(after=after) or [])
    rows: list[dict[str, Any]] = []
    run = run_id or str(uuid.uuid4())

    finalized_ids = set()
    submitted_ids = set()
    if not fills.empty:
        finalized_ids = set(
            fills.loc[fills["fill_type"].astype(str) == "stop", "order_id"].astype(str)
        )
        submitted_ids = set(
            fills.loc[
                fills["fill_type"].astype(str) == "stop_submitted", "order_id"
            ].astype(str)
        )

    for item in raw:
        order_type = str(item.get("order_type", "")).lower()
        order_id = str(item.get("order_id", ""))
        if not order_id:
            continue
        is_stop = ("stop" in order_type) or (order_id in submitted_ids)
        if not is_stop:
            continue
        if order_id in finalized_ids:
            continue
        qty = abs(float(item.get("qty", 0.0) or 0.0))
        if qty <= 0:
            continue
        filled_at = item.get("filled_at", _now_ts())
        rows.append(
            {
                "ts": _as_timestamp(filled_at),
                "strategy_id": strategy_id,
                "broker_id": broker_id,
                "ticker": str(item.get("ticker", "")).strip().upper(),
                "side": str(item.get("side", "sell")).lower(),
                "qty": qty,
                "price": item.get("price", pd.NA),
                "order_id": order_id,
                "fill_type": "stop",
                "run_id": run,
            }
        )

    if not rows:
        return fills
    return log_fills(rows, cache_dir=cache_dir)


def snapshot_mark_to_market(
    broker: Any,
    *,
    strategy_positions_map: dict[str, dict[str, float]] | None = None,
    as_of: Any | None = None,
    broker_id: str = BROKER_ALPACA,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """
    EOD snapshot: broker portfolio equity + per-sleeve virtual NAV.

    Sleeve equity compounds from prior sleeve equity using that sleeve's
    marked daily PnL (sum of position market-value changes). Allocation
    weights seed a sleeve only on its first equity row.
    """
    meta = get_meta(cache_dir)
    weights = _normalize_weights(meta.get("sleeve_weights") or DEFAULT_SLEEVE_WEIGHTS)
    stamp = _as_timestamp(as_of) if as_of is not None else _now_ts()
    day = stamp.normalize()

    account = broker.get_account()
    portfolio_equity = float(getattr(account, "equity"))
    cash = float(getattr(account, "cash", float("nan")))

    broker_positions = []
    if hasattr(broker, "get_positions_normalized"):
        broker_positions = list(broker.get_positions_normalized() or [])
    else:
        for pos in list(broker.get_positions() or []):
            broker_positions.append(
                {
                    "ticker": str(pos.symbol).strip().upper(),
                    "qty": float(pos.qty),
                    "avg_entry": float(getattr(pos, "avg_entry_price", float("nan"))),
                    "market_value": float(getattr(pos, "market_value", float("nan"))),
                    "unrealized_pl": float(
                        getattr(pos, "unrealized_pl", getattr(pos, "unrealized_plpc", float("nan")))
                    ),
                    "current_price": float(
                        getattr(pos, "current_price", getattr(pos, "lastday_price", float("nan")))
                    ),
                }
            )

    mark_by_ticker = {
        str(p["ticker"]).strip().upper(): p for p in broker_positions
    }

    if strategy_positions_map is None:
        qty_map = sleeve_qty_map(cache_dir=cache_dir)
        strategy_positions_map = {}
        for (sid, ticker), qty in qty_map.items():
            strategy_positions_map.setdefault(sid, {})[ticker] = qty

    # Ensure configured sleeves appear even when flat
    for sid in weights:
        strategy_positions_map.setdefault(sid, {})

    prev_equity = load_equity_daily(cache_dir)
    position_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = [
        {
            "date": day,
            "level": "portfolio",
            "strategy_id": "",
            "equity": portfolio_equity,
            "allocated_equity": portfolio_equity,
            "cash": cash,
        }
    ]

    for strategy_id, ticker_qty in strategy_positions_map.items():
        sleeve_mv = 0.0
        sleeve_upl = 0.0
        for ticker, qty in ticker_qty.items():
            q = float(qty)
            if abs(q) < 1e-12:
                continue
            mark = mark_by_ticker.get(str(ticker).strip().upper(), {})
            px = mark.get("current_price")
            if px is None or not (float(px) > 0):
                # Fall back to market_value / qty from broker when available
                broker_qty = float(mark.get("qty", 0.0) or 0.0)
                mv = mark.get("market_value")
                if mv is not None and abs(broker_qty) > 0:
                    px = abs(float(mv) / broker_qty)
                else:
                    px = float("nan")
            mv = float(q) * float(px) if px == px and float(px) > 0 else float("nan")
            avg_entry = mark.get("avg_entry", float("nan"))
            upl = mark.get("unrealized_pl", float("nan"))
            if mv == mv:
                sleeve_mv += mv
            if upl == upl and abs(float(mark.get("qty", 0.0) or 0.0) - q) < 1e-8:
                sleeve_upl += float(upl)
            position_rows.append(
                {
                    "ts": stamp,
                    "strategy_id": strategy_id,
                    "broker_id": broker_id,
                    "ticker": str(ticker).strip().upper(),
                    "qty": q,
                    "avg_entry": avg_entry,
                    "market_value": mv,
                    "unrealized_pl": upl,
                }
            )

        w = float(weights.get(strategy_id, 0.0))
        allocated = w * portfolio_equity
        prior = prev_equity.loc[
            (prev_equity["level"].astype(str) == "strategy")
            & (prev_equity["strategy_id"].astype(str) == str(strategy_id))
            & (pd.to_datetime(prev_equity["date"]) < day)
        ] if not prev_equity.empty else prev_equity

        if prior is not None and not prior.empty:
            prior_eq = float(prior.sort_values("date").iloc[-1]["equity"])
            # Compound with sleeve marked PnL when finite; else hold prior.
            if sleeve_upl == sleeve_upl:
                sleeve_equity = prior_eq + float(sleeve_upl)
            elif sleeve_mv == sleeve_mv:
                # Flat cash residual approximation inside allocated sleeve.
                sleeve_equity = prior_eq
            else:
                sleeve_equity = prior_eq
        else:
            # First observation: seed by allocation weight of portfolio equity.
            sleeve_equity = allocated

        equity_rows.append(
            {
                "date": day,
                "level": "strategy",
                "strategy_id": strategy_id,
                "equity": float(sleeve_equity),
                "allocated_equity": float(allocated),
                "cash": pd.NA,
            }
        )

    if position_rows:
        log_positions_snapshot(position_rows, cache_dir=cache_dir)
    log_equity_daily(equity_rows, cache_dir=cache_dir)

    return {
        "date": day,
        "portfolio_equity": portfolio_equity,
        "cash": cash,
        "positions": position_rows,
        "equity_rows": equity_rows,
    }


def backfill_equity_from_broker(
    broker: Any,
    *,
    period: str = "1M",
    timeframe: str = "1D",
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """
    Upsert portfolio (+ single-sleeve mirror) equity from broker daily history.

    Intended to repair sparse ``equity_daily`` ledgers when only Monday runner
    snapshots were written. Requires ``broker.get_portfolio_history``.
    When multiple sleeves are configured, only the portfolio level is filled;
    sleeve NAVs still need daily ``snapshot_mark_to_market``.
    """
    if not hasattr(broker, "get_portfolio_history"):
        raise TypeError("broker must implement get_portfolio_history")

    history = list(broker.get_portfolio_history(period=period, timeframe=timeframe) or [])
    if not history:
        return load_equity_daily(cache_dir)

    meta = get_meta(cache_dir)
    weights = _normalize_weights(meta.get("sleeve_weights") or DEFAULT_SLEEVE_WEIGHTS)
    single_sleeve = len(weights) == 1
    sole_sid = next(iter(weights)) if single_sleeve else None

    rows: list[dict[str, Any]] = []
    for item in history:
        day = _as_date(item["date"])
        equity = float(item["equity"])
        rows.append(
            {
                "date": day,
                "level": "portfolio",
                "strategy_id": "",
                "equity": equity,
                "allocated_equity": equity,
                "cash": pd.NA,
            }
        )
        if single_sleeve and sole_sid is not None:
            rows.append(
                {
                    "date": day,
                    "level": "strategy",
                    "strategy_id": sole_sid,
                    "equity": equity,
                    "allocated_equity": equity * float(weights[sole_sid]),
                    "cash": pd.NA,
                }
            )

    return log_equity_daily(rows, cache_dir=cache_dir)


def run_eod_snapshot(
    broker: Any,
    *,
    strategy_ids: list[str] | None = None,
    paper: bool = True,
    cache_dir: str | None = None,
    backfill: bool = False,
    backfill_period: str = "1M",
) -> dict[str, Any]:
    """
    Trading-day EOD entrypoint: sync stop fills, then mark portfolio + sleeves.

    ``paper`` is unused when an explicit ``broker`` is passed; kept for script
    callers that construct the broker themselves.
    When ``backfill`` is True, upsert daily equity from broker portfolio history
    before the live mark (repairs Monday-only ledger gaps).
    """
    _ = paper
    meta = get_meta(cache_dir)
    ids = list(strategy_ids or (meta.get("sleeve_weights") or DEFAULT_SLEEVE_WEIGHTS).keys())
    sync_summary = {}
    for sid in ids:
        before = len(load_fills(cache_dir))
        sync_stop_fills(broker, sid, cache_dir=cache_dir)
        after = len(load_fills(cache_dir))
        sync_summary[sid] = after - before

    backfill_rows = 0
    if backfill:
        before_eq = len(load_equity_daily(cache_dir))
        backfill_equity_from_broker(
            broker, period=backfill_period, cache_dir=cache_dir
        )
        backfill_rows = len(load_equity_daily(cache_dir)) - before_eq

    snap = snapshot_mark_to_market(broker, cache_dir=cache_dir)
    snap["stop_fills_added"] = sync_summary
    snap["backfill_rows_delta"] = backfill_rows
    return snap
