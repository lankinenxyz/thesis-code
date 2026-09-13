"""Rolling-window reporting (evaluation standard #3): no single cherry-picked window."""

from __future__ import annotations

import pandas as pd

from ..config import ROLLING_STEP, ROLLING_WINDOW
from . import metrics


def rolling_windows(r: pd.Series, window: int = ROLLING_WINDOW, step: int = ROLLING_STEP) -> pd.DataFrame:
    r = r.dropna()
    rows = []
    for start in range(0, len(r) - window + 1, step):
        chunk = r.iloc[start : start + window]
        rows.append({
            "start": chunk.index[0].date(),
            "end": chunk.index[-1].date(),
            "sharpe": metrics.sharpe(chunk),
            "total_return": float((1 + chunk).prod() - 1),
            "max_drawdown": metrics.max_drawdown(chunk),
        })
    return pd.DataFrame(rows)


def rolling_summary(r: pd.Series) -> dict:
    df = rolling_windows(r)
    if df.empty:
        return {"n_windows": 0}
    return {
        "n_windows": int(len(df)),
        "sharpe_median": float(df["sharpe"].median()),
        "sharpe_q25": float(df["sharpe"].quantile(0.25)),
        "sharpe_q75": float(df["sharpe"].quantile(0.75)),
        "sharpe_min": float(df["sharpe"].min()),
        "sharpe_max": float(df["sharpe"].max()),
        "share_windows_positive": float((df["total_return"] > 0).mean()),
    }
