"""HMM fit on train only; filter on val does not refit."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.processing.feature_implementation.hmm_regime import (
    filter_mr_probability,
    fit_gaussian_hmm_2state,
)


def test_hmm_fit_train_filter_val():
    rng = np.random.default_rng(1)
    train = pd.Series(rng.normal(0.0, 0.01, size=80))
    val = pd.Series(rng.normal(0.2, 0.05, size=40))
    params = fit_gaussian_hmm_2state(train, n_iter=15, seed=1)
    p_train = filter_mr_probability(train, params)
    p_val = filter_mr_probability(val, params)
    assert p_train.notna().any()
    assert p_val.notna().any()
    assert 0.0 <= float(p_val.dropna().iloc[-1]) <= 1.0
    # params are frozen: filtering val does not change means
    params2 = fit_gaussian_hmm_2state(train, n_iter=15, seed=1)
    assert params.means == params2.means
