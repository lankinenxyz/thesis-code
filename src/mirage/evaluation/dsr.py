"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, JPM 2014).

DSR = PSR(SR0): the probability that the observed Sharpe exceeds the expected
maximum Sharpe (SR0) obtainable from N unskilled trials, given non-normality
and track length. All Sharpe quantities here are PER-PERIOD (daily), not annualized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def prob_sharpe_ratio(sr_hat: float, sr0: float, n_obs: int, skew: float, kurt: float) -> float:
    """PSR: P[true SR > sr0]. `kurt` is NON-excess kurtosis (normal = 3)."""
    denom = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2)
    if denom == 0 or np.isnan(denom):
        return np.nan
    z = (sr_hat - sr0) * np.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(var_trials: float, n_trials: int) -> float:
    """E[max SR] across n_trials unskilled strategies with Sharpe variance var_trials."""
    if n_trials <= 1 or var_trials <= 0:
        return 0.0
    return float(
        np.sqrt(var_trials)
        * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    )


def deflated_sharpe(returns: pd.Series, trial_sharpes_daily: list[float], n_trials: int | None = None) -> dict:
    """DSR of `returns` given the daily Sharpe ratios of all trials attempted.

    `trial_sharpes_daily`: per-period Sharpe of every pre-registered variant
    (including this one) — its variance estimates V[SR] in the SR0 formula.
    """
    r = returns.dropna().astype(float)
    if len(r) < 20 or r.std(ddof=1) == 0:
        return {"dsr": np.nan, "sr0_daily": np.nan}
    sr_hat = float(r.mean() / r.std(ddof=1))
    n = n_trials or len(trial_sharpes_daily)
    var_trials = float(np.var(trial_sharpes_daily, ddof=1)) if len(trial_sharpes_daily) > 1 else 0.0
    sr0 = expected_max_sharpe(var_trials, n)
    return {
        "dsr": prob_sharpe_ratio(
            sr_hat, sr0, len(r), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
        ),
        "sr0_daily": sr0,
        "sr_hat_daily": sr_hat,
        "n_trials": n,
        "var_trials": var_trials,
    }


def daily_sharpe(returns: pd.Series) -> float:
    r = returns.dropna().astype(float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if len(r) > 2 and sd > 0 else np.nan
