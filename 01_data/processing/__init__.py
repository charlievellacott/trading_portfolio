"""Data processing package: cleaners and feature-store dispatchers."""

from data.processing.s1_feature_store import (
    add_beta_factors,
    add_gdelt_sentiment_factors,
    add_gk_vol_factors,
    add_gross_profitability_factors,
    add_idio_vol_factors,
    add_max_lottery_factors,
    add_near_52w_factors,
    add_obv_momentum_factors,
    add_short_flow_factors,
    add_size_value_factors,
    add_talib_factors,
    add_volume_factors,
)
from data.processing.s2_coint_store import (
    COINT_METRICS,
    build_pair_panel,
    compute_coint_metrics,
    compute_half_life,
    compute_kalman_hedge_spread,
    compute_spread_zscore,
    compute_static_hedge_spread,
    run_cointegration_test,
    screen_pair_cointegration,
)
from data.processing.s2_universe import (
    iter_same_venue_pairs,
    load_s2_universes,
    ticker_venue_key,
)

__all__ = [
    "COINT_METRICS",
    "add_beta_factors",
    "add_gdelt_sentiment_factors",
    "add_gk_vol_factors",
    "add_gross_profitability_factors",
    "add_idio_vol_factors",
    "add_max_lottery_factors",
    "add_near_52w_factors",
    "add_obv_momentum_factors",
    "add_short_flow_factors",
    "add_size_value_factors",
    "add_talib_factors",
    "add_volume_factors",
    "build_pair_panel",
    "compute_coint_metrics",
    "compute_half_life",
    "compute_kalman_hedge_spread",
    "compute_spread_zscore",
    "compute_static_hedge_spread",
    "iter_same_venue_pairs",
    "load_s2_universes",
    "run_cointegration_test",
    "screen_pair_cointegration",
    "ticker_venue_key",
]
