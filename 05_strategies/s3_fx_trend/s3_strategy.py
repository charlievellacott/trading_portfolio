"""S3 FX trend live/paper strategy: star stacks, close-t decision, open-t+1 fill."""

from __future__ import annotations

import os

import pandas as pd

from backtest.star_stack_io import load_star_stack
from data.repo_paths import repo_root
from strategies.base.strategy import Strategy
from strategies.s3_fx_trend.config import S3SimConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = repo_root()
DEFAULT_STAR_STACK_CORE = os.path.join(
    _REPO, "04_backtest", "s3_fx_trend", "artifacts", "s3_core_star_stack.json"
)
DEFAULT_STAR_STACK_EVENT = os.path.join(
    _REPO, "04_backtest", "s3_fx_trend", "artifacts", "s3_event_star_stack.json"
)


class S3Strategy(Strategy):
    """Frozen S3 core (+ optional event) star-stack wrapper.

    ``decide`` is an execution stub until the paper runner is wired; it returns
    an empty order list. Research simulators live in ``engine.py``.
    """

    def __init__(
        self,
        start_date: str | None = None,
        *,
        core_star_path: str | None = None,
        event_star_path: str | None = None,
        cfg: S3SimConfig | None = None,
    ) -> None:
        self.start_date = (
            pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
            if start_date
            else None
        )
        self.core_star_path = core_star_path or DEFAULT_STAR_STACK_CORE
        self.event_star_path = event_star_path
        self.stack: dict = {}
        self.event_stack: dict | None = None
        if os.path.isfile(self.core_star_path):
            self.stack = load_star_stack(self.core_star_path)
        if self.event_star_path and os.path.isfile(self.event_star_path):
            self.event_stack = load_star_stack(self.event_star_path)
        elif os.path.isfile(DEFAULT_STAR_STACK_EVENT):
            self.event_star_path = DEFAULT_STAR_STACK_EVENT
            self.event_stack = load_star_stack(DEFAULT_STAR_STACK_EVENT)

        if cfg is not None:
            self.cfg = cfg
        elif self.stack:
            # Deferred: backtest.s3_fx_trend.research may land after this package.
            from backtest.s3_fx_trend.research import config_from_stack

            self.cfg = config_from_stack(self.stack)
        else:
            self.cfg = S3SimConfig()

    def generate_features(self):
        raise NotImplementedError("S3 live feature build not wired yet")

    def generate_signal(self):
        raise NotImplementedError("S3 live signal not wired yet")

    def get_weights(self):
        raise NotImplementedError("S3 live weights not wired yet")

    def decide(self, current_date=None, panel=None) -> list:
        """Close-t decision stub → orders for open t+1. Not broker-wired yet."""
        _ = (current_date, panel, _HERE)
        return []
