# imports
import os
import joblib
import pandas as pd

from data.ingestion.alternative_data.fama_french_fetcher import fetch_ff_factors_daily
from data.ingestion.equity_fetcher import OHLCV_COLUMNS, fetch_ohlcv
from data.repo_paths import repo_root
from data.processing.cleaner import forward_fill_panel
from data.processing.feature_implementation.beta_features import market_return_frame
from data.processing.feature_implementation.momentum import add_raw_momentum
from data.processing.s1_feature_store import (
    add_beta_factors,
    add_gdelt_sentiment_factors,
    add_gross_profitability_factors,
    add_short_flow_factors,
    add_size_value_factors,
    add_volume_factors,
    drop_beta_workspace,
)
from strategies.base.strategy import Strategy
from risk.s1_equities.position_sizing import (
    monday_gross_leverage,
    monday_inv_vol_weights,
)
from risk.s1_equities.signal_conviction import parse_ic_scale_star
from risk.s1_equities.vol_targeting import parse_vol_target_star
from strategies.s1_equities.s1_utilities import (
    base_period_returns_from_cache,
    drop_non_model_columns,
    ensure_decision_date_ohlcv,
    ensure_opens_decision_date,
    fetch_opens_matrix,
    load_or_update_predictions_cache,
    load_prev_leverage,
    load_production_feature_cols,
    pit_safe_ic_inputs,
    rename_feature_stem,
    save_prev_leverage,
    to_s1_trade_date_panel,
)

# constants
_HERE = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_CSV = os.path.join(_HERE, "s1_universe.csv")
EXPECTED_N_TICKERS = 97
LIVE_HISTORY_START = "2020-01-01"
DEFAULT_MODEL_ARTIFACT_PATH = os.path.join(
    repo_root(),
    "03_models",
    "s1_equities",
    "model_artifacts",
    "s1_production_linear_model.joblib",
)
_SEC_FFILL_COLS = (
    "market_cap",
    "pe",
    "pb",
    "shares_outstanding",
    "book_equity",
    "eps_ttm",
)
_SEC_FFILL_LIMIT = 5

# class
class S1Strategy(Strategy):
    # Frozen OOS recipe (08 tearsheet)
    N_STAR = 15
    INV_VOL_WINDOW_STAR = 42
    STOP_PCT_STAR = 20  # percent adverse move; paper runner after fill (risk.pct_stop)
    VT_STAR = "vt_bayes_0.9_10_q0.75_db0.05"
    IC_STAR = "ic_k15_0.94_f0.25_c1.25"
    MAX_GROSS = 1.50
    BENCHMARK = "RSP"

    # constructor
    def __init__(self, start_date: str) -> None:
        as_of = pd.Timestamp(start_date).normalize()
        self.decision_date = as_of
        self.end_date = as_of.strftime("%Y-%m-%d")
        hist_start = pd.Timestamp(LIVE_HISTORY_START).normalize()
        if as_of < hist_start:
            self.start_date = as_of.strftime("%Y-%m-%d")
        else:
            self.start_date = LIVE_HISTORY_START

        self.tickers = self.get_tickers()
        
        self.model_artifact_path = DEFAULT_MODEL_ARTIFACT_PATH
        self.feature_cols: list[str] = []


    def generate_features(self) -> pd.DataFrame:
        # 1. Fetch OHLCV for the live universe
        frames = []
        for ticker in self.tickers:
            frames.append(fetch_ohlcv(ticker, self.start_date, self.end_date))

        if not frames:
            raise ValueError("No OHLCV frames fetched; tickers list is empty")

        data = pd.concat(frames, ignore_index=True)
        expected_cols = ["date", "ticker", *OHLCV_COLUMNS]
        missing_cols = [c for c in expected_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(f"OHLCV panel missing columns: {missing_cols}")

        data = data[expected_cols].sort_values(["date", "ticker"]).reset_index(drop=True)
        if data.empty:
            raise ValueError("OHLCV panel is empty after fetch")

        # Pre-open: dummy Monday bars so trade-date lag has today's decision row
        data = ensure_decision_date_ohlcv(data, self.decision_date, self.tickers)

        fetched = set(data["ticker"].astype(str).str.strip().str.upper().unique())
        missing_tickers = [t for t in self.tickers if t not in fetched]
        if missing_tickers:
            raise ValueError(
                f"OHLCV missing {len(missing_tickers)} universe ticker(s): "
                f"{missing_tickers[:10]}{'...' if len(missing_tickers) > 10 else ''}"
            )

        # 2. Trade-date panel: lag hlcv; feature_date = info cutoff for merges
        data = to_s1_trade_date_panel(data)

        # 3. Market / FF side frames (alt_data for beta_factors)
        bench = self.BENCHMARK.upper()
        market_ohlcv = fetch_ohlcv(bench, self.start_date, self.end_date)
        market_ohlcv = ensure_decision_date_ohlcv(
            market_ohlcv, self.decision_date, [bench]
        )
        market_returns = market_return_frame(market_ohlcv)
        ff_factors = fetch_ff_factors_daily(self.start_date, self.end_date)
        bench_kw = self.BENCHMARK.lower()

        # 4. Engineer production features (explicit calls; rename to joblib names)

        # raw_momentum_252_5
        data = add_raw_momentum(data, lookback=252, skip=5, col="raw_momentum")
        data = rename_feature_stem(data, "raw_momentum", "raw_momentum_252_5")

        # smart_residual_mom_189_42
        data = add_beta_factors(
            data,
            market_returns=market_returns,
            ff_factors=ff_factors,
            feature_subset=["smart_residual_mom"],
            formation_window=189,
            skip=42,
            benchmark=bench_kw,
        )
        data = rename_feature_stem(
            data, "smart_residual_mom", "smart_residual_mom_189_42"
        )

        # rel_downside_beta_252
        data = add_beta_factors(
            data,
            market_returns=market_returns,
            feature_subset=["rel_downside_beta"],
            windows=252,
            benchmark=bench_kw,
        )
        data = rename_feature_stem(
            data, "rel_downside_beta", "rel_downside_beta_252"
        )

        # rel_upside_beta_63
        data = add_beta_factors(
            data,
            market_returns=market_returns,
            feature_subset=["rel_upside_beta"],
            windows=63,
            benchmark=bench_kw,
        )
        data = rename_feature_stem(data, "rel_upside_beta", "rel_upside_beta_63")

        # smart_beta_hml_252
        data = add_beta_factors(
            data,
            market_returns=market_returns,
            ff_factors=ff_factors,
            feature_subset=["smart_beta_hml"],
            windows=252,
            benchmark=bench_kw,
        )
        data = rename_feature_stem(data, "smart_beta_hml", "smart_beta_hml_252")

        # downside_beta_42
        data = add_beta_factors(
            data,
            market_returns=market_returns,
            feature_subset=["downside_beta"],
            windows=42,
            benchmark=bench_kw,
        )
        data = rename_feature_stem(data, "downside_beta", "downside_beta_42")

        # upside_beta_42
        data = add_beta_factors(
            data,
            market_returns=market_returns,
            feature_subset=["upside_beta"],
            windows=42,
            benchmark=bench_kw,
        )
        data = rename_feature_stem(data, "upside_beta", "upside_beta_42")

        # market_corr (bare store name matches production)
        data = add_beta_factors(
            data,
            market_returns=market_returns,
            feature_subset=["market_corr"],
            windows=252,
            benchmark=bench_kw,
        )

        # beta_mkt_interact
        data = add_beta_factors(
            data,
            market_returns=market_returns,
            feature_subset=["beta_mkt_interact"],
            windows=252,
            mkt_horizon=5,
            benchmark=bench_kw,
        )
        data = drop_beta_workspace(data)

        # size_mom_126 — first SEC size/value fetch
        data = add_size_value_factors(
            data,
            feature_subset=["size_mom"],
            window=126,
            size_value_data_exists=False,
        )
        data = rename_feature_stem(data, "size_mom", "size_mom_126")
        sec_present = [c for c in _SEC_FFILL_COLS if c in data.columns]
        if sec_present:
            data = data.sort_values(["ticker", "date"])
            data[sec_present] = data.groupby("ticker", sort=False)[sec_present].ffill(
                limit=_SEC_FFILL_LIMIT
            )

        # val_roc_pb_252 / val_roc_pe_252
        data = add_size_value_factors(
            data,
            feature_subset=["val_roc_pb"],
            window=252,
            size_value_data_exists=True,
        )
        data = rename_feature_stem(data, "val_roc_pb", "val_roc_pb_252")
        data = add_size_value_factors(
            data,
            feature_subset=["val_roc_pe"],
            window=252,
            size_value_data_exists=True,
        )
        data = rename_feature_stem(data, "val_roc_pe", "val_roc_pe_252")

        # log_mcap
        data = add_size_value_factors(
            data,
            feature_subset=["log_mcap"],
            size_value_data_exists=True,
        )

        # val_mom_dist_252_21
        data = add_size_value_factors(
            data,
            feature_subset=["val_mom_dist"],
            mom_lookback=252,
            mom_skip=21,
            size_value_data_exists=True,
        )
        data = rename_feature_stem(data, "val_mom_dist", "val_mom_dist_252_21")

        # gross_profitability
        data = add_gross_profitability_factors(
            data,
            feature_subset=["gross_profitability"],
            gross_profitability_data_exists=False,
        )

        # filing_clock_expected_until
        data = add_short_flow_factors(
            data,
            feature_subset=["filing_expected_until"],
            filing_clock_data_exists=False,
        )

        # short_flow_ratio
        data = add_short_flow_factors(
            data,
            feature_subset=["ratio"],
            short_volume_data_exists=False,
        )

        # abnormal_volume
        data = add_volume_factors(
            data,
            feature_subset=["abnormal_volume"],
            smooth_window=5,
            baseline_window=60,
        )

        # gdelt_tone_x_attention_21 — first GDELT fetch
        data = add_gdelt_sentiment_factors(
            data,
            feature_subset=["tone_x_attention"],
            window=21,
            sentiment_data_exists=False,
        )
        data = rename_feature_stem(
            data, "gdelt_tone_x_attention", "gdelt_tone_x_attention_21"
        )

        # gdelt_attention_5
        data = add_gdelt_sentiment_factors(
            data,
            feature_subset=["attention"],
            window=5,
            sentiment_data_exists=True,
        )
        data = rename_feature_stem(data, "gdelt_attention", "gdelt_attention_5")

        # 5. Align to production feature_cols and drop workspace cols
        feature_cols = load_production_feature_cols(self.model_artifact_path)
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            raise ValueError(
                f"engineered panel missing production feature_cols: {missing}"
            )
        self.feature_cols = list(feature_cols)
        data = drop_non_model_columns(data, feature_cols)
        return data

    def generate_signal(self, panel: pd.DataFrame = None) -> pd.Series:
        # 1. Feature panel
        if panel is None:
            panel = self.generate_features()
        # 2. Temporary prediction column (cache + get_weights only)
        model = self.load_production_model()
        feature_cols = self.feature_cols or load_production_feature_cols(
            self.model_artifact_path
        )
        work = panel.copy()
        # Slim-ffill before predict (production parity)
        work = forward_fill_panel(work, columns=feature_cols, limit=None)
        complete = work[feature_cols].notna().all(axis=1)
        work = work.loc[complete].copy()
        work["prediction"] = model.predict(work[feature_cols])
        # 3. Update predictions cache (creates if missing)
        load_or_update_predictions_cache(self, work.rename(columns={"prediction": "score"}))
        # 4. Replace prediction with signed weight
        panel_w = self.get_weights(work)
        # 5. Return non-zero signed weights (ticker-indexed)
        latest = pd.Timestamp(panel_w["date"].max())
        day = panel_w.loc[panel_w["date"] == latest].set_index("ticker")["weight"]
        return day.astype(float).replace(0.0, pd.NA).dropna()

    def get_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        # 1. Ensure predictions cache is current
        scored = panel.copy()
        if "score" not in scored.columns and "prediction" in scored.columns:
            scored = scored.rename(columns={"prediction": "score"})
        preds = load_or_update_predictions_cache(self, scored)

        # Decision date = this Monday (pre-open placeholder rows if needed)
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        decision_date = pd.Timestamp(self.decision_date).normalize()
        preds = preds.copy()
        preds["date"] = pd.to_datetime(preds["date"]).dt.normalize()
        day = preds.loc[preds["date"] == decision_date]
        if day.empty and "score" in scored.columns:
            scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
            day = scored.loc[scored["date"] == decision_date]
        if day.empty:
            raise ValueError(
                f"no scores for decision_date={decision_date.date()}"
            )
        scores = day.set_index("ticker")["score"].astype(float)
        scores.index = pd.Index(scores.index.astype(str).str.strip().str.upper())

        # 2. Fetch opens for inv_vol window + recent holds
        opens = fetch_opens_matrix(self.tickers, self.start_date, self.end_date)
        opens = ensure_opens_decision_date(opens, decision_date)

        # 3. Signed inv-vol base book (+ long / - short)
        w = monday_inv_vol_weights(
            scores,
            opens,
            decision_date=decision_date,
            n=self.N_STAR,
            window=self.INV_VOL_WINDOW_STAR,
        )

        # 4. PIT-safe history for leverage overlay
        past_returns = base_period_returns_from_cache(
            preds,
            opens,
            decision_date=decision_date,
            n=self.N_STAR,
            inv_vol_window=self.INV_VOL_WINDOW_STAR,
            pit_lag=1,
        )
        ic_hist = pit_safe_ic_inputs(preds, decision_date, pit_lag=1)
        prev_lev = load_prev_leverage()

        # 5. Gross leverage scalar (vol target x Bayesian IC)
        vt_cfg = parse_vol_target_star(self.VT_STAR)
        ic_cfg = parse_ic_scale_star(self.IC_STAR)
        sizing = monday_gross_leverage(
            past_returns,
            vt_cfg,
            past_ic=ic_hist["ic"] if len(ic_hist) else None,
            past_n_names=ic_hist["n_names"] if len(ic_hist) else None,
            ic_cfg=ic_cfg,
            prev_leverage=prev_lev,
            max_gross=self.MAX_GROSS,
        )
        save_prev_leverage(sizing["leverage"])

        # 6. Signed levered weights
        if w.empty:
            w_lev = w
        else:
            w_lev = w * float(sizing["leverage"])

        # 7. Keep score for the runner log; add weight (zeros for non-selected)
        out = panel.loc[panel["date"] == decision_date].copy()
        if out.empty:
            raise ValueError(
                f"no panel rows for decision_date={decision_date.date()}"
            )
        if "prediction" in out.columns:
            out = out.drop(columns=["prediction"])
        out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
        out["score"] = out["ticker"].map(scores)
        out["weight"] = 0.0
        if not w_lev.empty:
            mapped = out["ticker"].map(w_lev)
            out["weight"] = mapped.fillna(0.0).astype(float)
        out = out.reset_index(drop=True)
        # Note: 'out' is a DataFrame with one row per ticker for the decision date,
        # columns include at least ['date', 'ticker', ..., 'weight'], where 'weight'
        # is the final signed portfolio weight for each ticker (0.0 for non-selected,
        # possibly positive/negative, depending on the direction, for selected names).
        return out

    def get_tickers(self) -> list[str]:
        if not os.path.isfile(UNIVERSE_CSV):
            raise FileNotFoundError(UNIVERSE_CSV)

        tickers = (
            pd.read_csv(UNIVERSE_CSV)["ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
        if len(tickers) != EXPECTED_N_TICKERS:
            raise ValueError(
                f"Expected {EXPECTED_N_TICKERS} tickers in {UNIVERSE_CSV}, "
                f"got {len(tickers)}"
            )
        return tickers

    def load_production_model(self):
        # Production artifact is a dict: {"model": Ridge, "feature_cols": [...], ...}
        if not os.path.isfile(self.model_artifact_path):
            raise FileNotFoundError(self.model_artifact_path)
        payload = joblib.load(self.model_artifact_path)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(
                f"{self.model_artifact_path!r} must be a dict with key 'model'"
            )
        if "feature_cols" in payload:
            self.feature_cols = [str(c) for c in payload["feature_cols"]]
        return payload["model"]
