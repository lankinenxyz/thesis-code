"""Regime labeling (evaluation standard #5): bull/bear by SPY vs 200-day MA;
high/low volatility by ^VIX median split over the evaluation window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def label_regimes(spy: pd.Series, vix: pd.Series, eval_index: pd.DatetimeIndex) -> pd.DataFrame:
    ma200 = spy.rolling(200).mean()
    trend = pd.Series(np.where(spy > ma200, "bull", "bear"), index=spy.index)
    vix_eval = vix.reindex(eval_index).ffill()
    vol = pd.Series(
        np.where(vix_eval > vix_eval.median(), "high_vol", "low_vol"), index=eval_index
    )
    return pd.DataFrame({
        "trend": trend.reindex(eval_index).ffill(),
        "vol": vol,
    })


def regime_breakdown(r: pd.Series, regimes: pd.DataFrame) -> dict:
    r = r.dropna()
    out = {}
    joined = pd.concat([r.rename("r"), regimes], axis=1).dropna(subset=["r"])
    for col in ("trend", "vol"):
        for label, grp in joined.groupby(col):
            out[f"{col}:{label}"] = {
                "n_days": int(len(grp)),
                "sharpe": metrics.sharpe(grp["r"]),
                "mean_daily_bps": float(grp["r"].mean() * 1e4),
                "total_return": float((1 + grp["r"]).prod() - 1),
            }
    return out
