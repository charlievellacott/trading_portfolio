"""S1 weekly → weekday expansion for FTMO daily rules."""

from __future__ import annotations

import pandas as pd

from risk.analytics.prop_firm.s1_calendar import weekly_to_weekday_returns


def test_weekly_ret_lands_on_monday_other_weekdays_zero():
    monday = pd.Timestamp("2020-01-06")  # Monday
    weekly = pd.Series([0.10], index=pd.DatetimeIndex([monday]), name="ret")
    daily = weekly_to_weekday_returns(weekly)
    assert daily.loc[monday] == 0.10
    weekdays = pd.bdate_range(monday, monday + pd.offsets.BDay(4))
    assert list(weekdays) == list(daily.index)
    assert (daily.iloc[1:] == 0.0).all()


def test_compound_matches_weekly():
    idx = pd.DatetimeIndex(["2020-01-06", "2020-01-13"])  # two Mondays
    weekly = pd.Series([0.05, -0.02], index=idx, name="s1")
    daily = weekly_to_weekday_returns(weekly)
    w1 = (1.0 + daily.loc["2020-01-06":"2020-01-10"]).prod() - 1.0
    w2 = (1.0 + daily.loc["2020-01-13":"2020-01-17"]).prod() - 1.0
    assert abs(w1 - 0.05) < 1e-12
    assert abs(w2 - (-0.02)) < 1e-12
    assert abs(float((1.0 + daily).prod() - 1.0) - float((1.0 + weekly).prod() - 1.0)) < 1e-12
