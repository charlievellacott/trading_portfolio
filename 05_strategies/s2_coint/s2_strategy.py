"""S2 cointegration live strategy: frozen STAR book, close-t signal, open-t+1 fill."""

from __future__ import annotations

import pandas as pd

from backtest.s2_coint.report import load_star_stack
from backtest.s2_coint.research import DEFAULT_STAR_STACK, config_from_stack
from data.ingestion.equity_fetcher import OHLCV_COLUMNS, fetch_ohlcv
from data.processing.s2_coint_store import build_pair_panel
from strategies.base.strategy import Strategy
from strategies.s2_coint.live_decision import walk_live_book
from strategies.s2_coint.s2_utilities import (
    long_ohlcv_to_frames,
    parse_pair_id,
    save_book_returns_cache,
    save_panel_cache,
    save_sizing_state,
    tickers_from_pair_ids,
)

LIVE_HISTORY_START = "2018-01-01"


class S2Strategy(Strategy):
    """Hardcoded Universe D freeze book from ``s2_star_stack.json``."""

    def __init__(
        self,
        start_date: str,
        *,
        star_stack_path: str | None = None,
    ) -> None:
        fill = pd.Timestamp(start_date).normalize()
        self.fill_date = fill
        self.end_date = fill.strftime("%Y-%m-%d")
        hist_start = pd.Timestamp(LIVE_HISTORY_START).normalize()
        self.start_date = (
            fill.strftime("%Y-%m-%d") if fill < hist_start else LIVE_HISTORY_START
        )

        path = star_stack_path or DEFAULT_STAR_STACK
        self.star_stack_path = path
        self.stack = load_star_stack(path)
        self.cfg = config_from_stack(self.stack)
        self.pair_ids = [str(p) for p in (self.stack.get("PAIRS_STAR") or [])]
        if not self.pair_ids:
            raise ValueError(f"PAIRS_STAR missing in {path}")
        self.pairs = [parse_pair_id(pid) for pid in self.pair_ids]
        self.tickers = tickers_from_pair_ids(self.pair_ids)
        self._last_book = None

    def generate_features(self) -> pd.DataFrame:
        frames = []
        for ticker in self.tickers:
            frames.append(
                fetch_ohlcv(
                    ticker,
                    self.start_date,
                    self.end_date,
                    auto_adjust=False,
                )
            )
        if not frames:
            raise ValueError("No OHLCV frames fetched; tickers list is empty")
        data = pd.concat(frames, ignore_index=True)
        expected = ["date", "ticker", *OHLCV_COLUMNS]
        missing = [c for c in expected if c not in data.columns]
        if missing:
            raise ValueError(f"OHLCV panel missing columns: {missing}")
        data = data[expected].copy()
        data["date"] = pd.to_datetime(data["date"]).dt.normalize()
        data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
        # Fill morning: never use a same-session incomplete bar as close t.
        data = data.loc[data["date"] < self.fill_date].copy()
        if data.empty:
            raise ValueError(
                f"OHLCV empty after dropping fill_date {self.fill_date.date()} and later"
            )
        fetched = set(data["ticker"].unique())
        missing_tickers = [t for t in self.tickers if t not in fetched]
        if missing_tickers:
            raise ValueError(f"OHLCV missing ticker(s): {missing_tickers}")

        ohlc = long_ohlcv_to_frames(data)
        panel = build_pair_panel(
            ohlc,
            self.pairs,
            ols_window=int(self.cfg.ols_window),
            z_window=int(self.cfg.z_window),
            hl_window=int(self.cfg.hl_window),
            include_adf_pvalue=True,
            include_variance_jump=True,
            hedge=str(self.cfg.hedge),
        )
        if panel.empty:
            raise ValueError("pair panel is empty after build_pair_panel")
        save_panel_cache(panel)
        return panel

    def generate_signal(self, panel: pd.DataFrame | None = None) -> pd.Series:
        weights = self.get_weights(panel)
        s = weights.set_index("ticker")["weight"].astype(float)
        return s.loc[s != 0.0]

    def get_weights(self, panel: pd.DataFrame | None = None) -> pd.DataFrame:
        if panel is None:
            panel = self.generate_features()
        work = panel.copy()
        work["date"] = pd.to_datetime(work["date"]).dt.normalize()
        work = work.loc[work["date"] < self.fill_date].copy()
        if work.empty:
            raise ValueError(
                f"no signal bars before fill_date {self.fill_date.date()}"
            )
        asof = pd.Timestamp(work["date"].max()).normalize()
        book = walk_live_book(
            work,
            self.cfg,
            asof=asof,
            universe_tickers=self.tickers,
        )
        self._last_book = book
        save_book_returns_cache(book.unlevered_returns)
        save_sizing_state(
            {
                "fill_date": self.fill_date.strftime("%Y-%m-%d"),
                "signal_date": pd.Timestamp(book.signal_date).strftime("%Y-%m-%d"),
                "prev_leverage": float(book.leverage),
                "open_pairs": {
                    pid: {
                        "side": int(st["side"]),
                        "scale": float(st["scale"]),
                        "entry_date": str(st.get("entry_date", "")),
                    }
                    for pid, st in book.open_pos.items()
                },
                "z_window": int(self.cfg.z_window),
                "mean_abs_mode": "rolling_abs_z",
            }
        )
        return book.weights
