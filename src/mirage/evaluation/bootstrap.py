"""Stationary block bootstrap (Politis & Romano 1994) for Sharpe-difference inference.

Both strategies are resampled with the SAME index sequence, preserving their
cross-correlation — the quantity bootstrapped is the paired Sharpe difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=np.int64)
    p = 1.0 / mean_block
    i = int(rng.integers(n))
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = int(rng.integers(n))
        else:
            i = (i + 1) % n
    return idx


def _sharpe_np(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


def sharpe_diff_test(
    r_a: pd.Series,
    r_b: pd.Series,
    n_resamples: int = 10_000,
    mean_block: float = 10.0,
    seed: int = 0,
) -> dict:
    """Bootstrap CI and two-sided p-value for daily Sharpe(a) - Sharpe(b)."""
    joined = pd.concat([r_a, r_b], axis=1, keys=["a", "b"]).dropna()
    a, b = joined["a"].to_numpy(float), joined["b"].to_numpy(float)
    n = len(a)
    if n < 40:
        return {"diff_daily": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan, "n_obs": n}
    observed = _sharpe_np(a) - _sharpe_np(b)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        diffs[k] = _sharpe_np(a[idx]) - _sharpe_np(b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff_daily": observed,
        "diff_annualized": observed * np.sqrt(252),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_lo_annualized": float(lo * np.sqrt(252)),
        "ci_hi_annualized": float(hi * np.sqrt(252)),
        "p_value": float(min(1.0, p)),
        "n_obs": n,
        "n_resamples": n_resamples,
        "mean_block": mean_block,
    }
