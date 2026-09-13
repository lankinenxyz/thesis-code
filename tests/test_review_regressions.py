"""Regression tests for defects found in the adversarial code review."""

import pandas as pd
import pytest

from mirage.backtest.engine import run_backtest
from mirage.backtest.view import MarketData
from mirage.data.news import close_utc
from mirage.evaluation.metrics import max_drawdown
from mirage.strategies.base import Decision, Strategy

IDX3 = pd.bdate_range("2025-01-06", periods=3)


def test_mdd_drawdown_from_first_observation():
    # Wealth path 0.80 -> 1.00: true MDD is -20% from initial capital.
    r = pd.Series([-0.20, 0.25], index=IDX3[:2])
    assert max_drawdown(r) == pytest.approx(-0.20, rel=1e-9)


def test_mdd_still_correct_for_later_drawdown():
    r = pd.Series([0.10, -0.50, 0.10], index=IDX3)
    assert max_drawdown(r) == pytest.approx(-0.50, rel=1e-9)


def test_early_close_news_cutoff():
    # 2025-11-28 was a 13:00-ET half day: close must be 18:00 UTC, not 21:00.
    assert close_utc("2025-11-28") == pd.Timestamp("2025-11-28 18:00", tz="UTC")
    assert close_utc("2025-11-26") == pd.Timestamp("2025-11-26 21:00", tz="UTC")


class Monthly5050(Strategy):
    name = "m5050"
    rebalance = "monthly"

    def decide(self, t, view):
        return Decision({"A": 0.5, "B": 0.5})


def test_no_spurious_month_end_in_truncated_window():
    # Full calendar covers all of Jan+Feb; window ends mid-Feb. The mid-Feb
    # last window day must NOT be treated as a month-end rebalance.
    cal = pd.bdate_range("2025-01-01", "2025-02-28")
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=cal)
    market = MarketData(prices)
    res = run_backtest(Monthly5050(), market, "2025-01-01", "2025-02-13", cost_grid=(25,))
    dec_days = list(res.decisions["decision_day"])
    assert pd.Timestamp("2025-02-13") not in dec_days
    assert pd.Timestamp("2025-01-31") in dec_days


class Daily5050(Monthly5050):
    name = "d5050"
    rebalance = "daily"


def test_no_decision_on_final_window_day():
    cal = pd.bdate_range("2025-01-06", periods=5)
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=cal)
    market = MarketData(prices)
    res = run_backtest(Daily5050(), market, cal[0], cal[-1], cost_grid=(25,))
    assert cal[-1] not in list(res.decisions["decision_day"])
    assert res.returns.loc[cal[-1], "turnover"] == 0.0
