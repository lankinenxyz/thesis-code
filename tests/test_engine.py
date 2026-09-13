import numpy as np
import pandas as pd
import pytest

from mirage.backtest.engine import run_backtest
from mirage.backtest.view import MarketData
from mirage.strategies.base import Decision, Strategy


def make_market():
    idx = pd.bdate_range("2025-01-06", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 121.0], "B": [100.0, 100.0, 100.0, 110.0]}, index=idx
    )
    return MarketData(prices), idx


class OnceAll_A(Strategy):
    name = "once_a"
    rebalance = "once"

    def decide(self, t, view):
        return Decision({"A": 1.0})


class Rebalance5050(Strategy):
    name = "rb5050"
    rebalance = "daily"

    def decide(self, t, view):
        return Decision({"A": 0.5, "B": 0.5})


def test_once_strategy_returns_and_costs():
    market, idx = make_market()
    res = run_backtest(OnceAll_A(), market, idx[0], idx[-1], cost_grid=(0, 10))
    r = res.returns
    # day0: decision at close, no return yet; entering costs turnover=1
    assert r.loc[idx[0], "gross"] == 0.0
    assert r.loc[idx[0], "turnover"] == pytest.approx(1.0)
    assert r.loc[idx[0], "net_10"] == pytest.approx(-10 / 1e4)
    # day1, day2: A gains 10% each
    assert r.loc[idx[1], "gross"] == pytest.approx(0.10)
    assert r.loc[idx[2], "gross"] == pytest.approx(0.10)
    # day3: A flat
    assert r.loc[idx[3], "gross"] == pytest.approx(0.0)
    # net_0 equals gross everywhere
    assert np.allclose(r["net_0"], r["gross"])


def test_drift_and_turnover_hand_computed():
    market, idx = make_market()
    res = run_backtest(Rebalance5050(), market, idx[0], idx[2], cost_grid=(0,))
    r = res.returns
    # day0: enter 50/50, turnover = 1.0
    assert r.loc[idx[0], "turnover"] == pytest.approx(1.0)
    # day1: rA=10%, rB=0 -> portfolio +5%; drifted weights A=0.55/1.05, B=0.5/1.05
    assert r.loc[idx[1], "gross"] == pytest.approx(0.05)
    wA = 0.5 * 1.10 / 1.05
    wB = 0.5 * 1.00 / 1.05
    expected_turnover = abs(0.5 - wA) + abs(0.5 - wB)
    assert r.loc[idx[1], "turnover"] == pytest.approx(expected_turnover, rel=1e-9)


def test_missing_return_liquidates_to_cash():
    idx = pd.bdate_range("2025-01-06", periods=3)
    prices = pd.DataFrame({"A": [100.0, np.nan, np.nan], "B": [100.0, 110.0, 110.0]}, index=idx)
    market = MarketData(prices)

    class HoldA(Strategy):
        name = "hold_a"
        rebalance = "once"

        def decide(self, t, view):
            return Decision({"A": 1.0})

    res = run_backtest(HoldA(), market, idx[0], idx[-1], cost_grid=(0,))
    # A has no return on day1 -> liquidated at 0 return; day2 also 0
    assert res.returns["gross"].iloc[1] == 0.0
    assert res.returns["gross"].iloc[2] == 0.0


def test_decision_log_populated():
    market, idx = make_market()
    res = run_backtest(OnceAll_A(), market, idx[0], idx[-1], cost_grid=(0,))
    assert len(res.decisions) == 1
    assert res.decisions.iloc[0]["n_positions"] == 1


def test_backtest_resume_from_checkpoint(tmp_path):
    market, idx = make_market()
    checkpoint = tmp_path / "checkpoint.pkl"

    class InterruptOnce(Strategy):
        name = "interrupt_once"
        rebalance = "daily"

        def __init__(self, fail_after: int | None):
            self.fail_after = fail_after
            self.n = 0

        def state_dict(self):
            return {"n": self.n}

        def load_state_dict(self, state):
            self.n = state.get("n", 0)

        def decide(self, t, view):
            if self.fail_after is not None and self.n >= self.fail_after:
                raise RuntimeError("intentional interruption")
            self.n += 1
            return Decision({"A": 0.5, "B": 0.5})

    with pytest.raises(RuntimeError):
        run_backtest(InterruptOnce(fail_after=1), market, idx[0], idx[-1], cost_grid=(0,), checkpoint_path=checkpoint)
    assert checkpoint.exists()

    resumed = run_backtest(InterruptOnce(fail_after=None), market, idx[0], idx[-1], cost_grid=(0,), checkpoint_path=checkpoint)
    full = run_backtest(InterruptOnce(fail_after=None), market, idx[0], idx[-1], cost_grid=(0,))

    pd.testing.assert_frame_equal(resumed.returns, full.returns)
    pd.testing.assert_frame_equal(resumed.weights, full.weights)
