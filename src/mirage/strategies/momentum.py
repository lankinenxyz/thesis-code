"""B2: cross-sectional 12-1 momentum (Jegadeesh-Titman style), the cheap non-LLM brain.

At each month-end: rank point-in-time members by return from t-252 to t-21
(skipping the most recent month); long the top decile, equal-weighted; hold one month.
"""

from __future__ import annotations

import math

import pandas as pd

from ..config import MOM_LOOKBACK, MOM_MIN_OBS, MOM_SKIP, MOM_TOP_FRACTION
from .base import Decision, Strategy


class Momentum(Strategy):
    name = "B2_momentum_12_1"
    rebalance = "monthly"

    def decide(self, t, view) -> Decision | None:
        p = view.prices
        if len(p) <= MOM_LOOKBACK:
            return None
        p_now = p.iloc[-1]
        p_skip = p.iloc[-1 - MOM_SKIP]
        p_lb = p.iloc[-1 - MOM_LOOKBACK]
        window = p.iloc[-1 - MOM_LOOKBACK : ]

        valid = p_now.notna() & p_skip.notna() & p_lb.notna() & (window.notna().sum() >= MOM_MIN_OBS)
        if view.members is not None:
            valid &= pd.Series(p.columns.isin(set(view.members)), index=p.columns)
        if "SPY" in valid.index:
            valid["SPY"] = False
        score = (p_skip / p_lb - 1.0)[valid]
        if len(score) < 20:
            return None
        k = max(1, math.ceil(len(score) * MOM_TOP_FRACTION))
        top = score.nlargest(k)
        w = 1.0 / k
        return Decision({tk: w for tk in top.index})
