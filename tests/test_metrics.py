import numpy as np
import pandas as pd
import pytest

from mirage.evaluation import metrics
from mirage.evaluation.dsr import deflated_sharpe, expected_max_sharpe
from mirage.evaluation.bootstrap import sharpe_diff_test

IDX = pd.bdate_range("2025-01-01", periods=252)


def test_cagr_known():
    r = pd.Series(0.001, index=IDX)  # 10bps/day for exactly one trading year
    assert metrics.cagr(r) == pytest.approx(1.001**252 - 1, rel=1e-9)


def test_sharpe_known():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 252), index=IDX)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe(r) == pytest.approx(expected, rel=1e-9)


def test_max_drawdown_known_path():
    # +10%, -50%, +10%: peak after day1, trough after day2 -> MDD = -50%
    r = pd.Series([0.10, -0.50, 0.10], index=IDX[:3])
    assert metrics.max_drawdown(r) == pytest.approx(-0.50, rel=1e-9)


def test_sortino_no_downside_is_nan():
    r = pd.Series(0.001, index=IDX)
    assert np.isnan(metrics.sortino(r))


def test_nw_tstat_zero_mean_small():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 500), index=pd.bdate_range("2024-01-01", periods=500))
    assert abs(metrics.newey_west_tstat(r)) < 3


def test_dsr_bounds_and_monotonicity():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, 252), index=IDX)
    trials = [0.02, 0.05, -0.01, 0.03]
    d_small = deflated_sharpe(r, trials, n_trials=2)
    d_big = deflated_sharpe(r, trials, n_trials=100)
    assert 0 <= d_small["dsr"] <= 1 and 0 <= d_big["dsr"] <= 1
    assert d_big["dsr"] <= d_small["dsr"]  # more trials -> harder to look skilled
    assert expected_max_sharpe(0.0, 10) == 0.0
    assert expected_max_sharpe(0.01, 100) > expected_max_sharpe(0.01, 10)


def test_bootstrap_identical_series_p_high():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 300), index=pd.bdate_range("2024-01-01", periods=300))
    t = sharpe_diff_test(r, r.copy(), n_resamples=200, seed=1)
    assert t["diff_daily"] == pytest.approx(0.0, abs=1e-12)
    assert t["p_value"] > 0.5


def test_bootstrap_detects_large_difference():
    idx = pd.bdate_range("2024-01-01", periods=400)
    rng = np.random.default_rng(4)
    good = pd.Series(rng.normal(0.002, 0.005, 400), index=idx)   # huge Sharpe
    bad = pd.Series(rng.normal(-0.002, 0.005, 400), index=idx)
    t = sharpe_diff_test(good, bad, n_resamples=500, seed=2)
    assert t["p_value"] < 0.05 and t["ci_lo"] > 0
