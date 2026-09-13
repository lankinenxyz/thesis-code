"""Performance metrics with HAC (Newey-West) inference for the mean return."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import RISK_FREE_ANNUAL, TRADING_DAYS


def _clean(r: pd.Series) -> pd.Series:
    return r.dropna().astype(float)


def cagr(r: pd.Series) -> float:
    r = _clean(r)
    if len(r) == 0:
        return np.nan
    wealth = float((1 + r).prod())
    if wealth <= 0:
        return -1.0
    return wealth ** (TRADING_DAYS / len(r)) - 1


def ann_vol(r: pd.Series) -> float:
    r = _clean(r)
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 2 else np.nan


def sharpe(r: pd.Series) -> float:
    r = _clean(r) - RISK_FREE_ANNUAL / TRADING_DAYS
    sd = r.std(ddof=1)
    if len(r) < 3 or sd == 0 or np.isnan(sd):
        return np.nan
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def newey_west_tstat(r: pd.Series, lags: int | None = None) -> float:
    """HAC t-statistic for H0: mean excess daily return = 0."""
    r = _clean(r) - RISK_FREE_ANNUAL / TRADING_DAYS
    if len(r) < 10:
        return np.nan
    if lags is None:
        lags = int(np.floor(1.5 * len(r) ** (1 / 3)))
    x = np.ones((len(r), 1))
    fit = sm.OLS(r.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def sortino(r: pd.Series) -> float:
    r = _clean(r) - RISK_FREE_ANNUAL / TRADING_DAYS
    downside = r[r < 0]
    if len(r) < 3 or len(downside) == 0:
        return np.nan
    dd = np.sqrt((downside ** 2).sum() / len(r))
    if dd == 0:
        return np.nan
    return float(r.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(r: pd.Series) -> float:
    r = _clean(r)
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    # Running peak must include the initial capital of 1.0, otherwise a drawdown
    # that begins at the first observation is missed (review finding, fixed).
    peak = wealth.cummax().clip(lower=1.0)
    return float((wealth / peak - 1).min())


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    if mdd is np.nan or mdd == 0:
        return np.nan
    return float(cagr(r) / abs(mdd))


def hit_rate(r: pd.Series) -> float:
    r = _clean(r)
    traded = r[r != 0]
    return float((traded > 0).mean()) if len(traded) else np.nan


def annual_turnover(turnover: pd.Series) -> float:
    t = _clean(turnover)
    if len(t) == 0:
        return np.nan
    return float(t.sum() * TRADING_DAYS / len(t))


def summary(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    out = {
        "n_days": int(len(_clean(r))),
        "total_return": float((1 + _clean(r)).prod() - 1) if len(_clean(r)) else np.nan,
        "cagr": cagr(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r),
        "nw_tstat": newey_west_tstat(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r),
        "hit_rate": hit_rate(r),
    }
    if turnover is not None:
        out["ann_turnover"] = annual_turnover(turnover)
    return out
