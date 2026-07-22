"""Compatibility shim — Fama-French fetcher lives under ``alternative_data``."""

from data.ingestion.alternative_data.fama_french_fetcher import fetch_ff_factors_daily

__all__ = ["fetch_ff_factors_daily"]
