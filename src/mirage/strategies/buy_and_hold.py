"""B1: buy-and-hold SPY. B1e: monthly-rebalanced equal-weight point-in-time universe."""

from __future__ import annotations

import pandas as pd

from .base import Decision, Strategy


class BuyAndHold(Strategy):
    name = "B1_spy_buy_and_hold"
    rebalance = "once"

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker

    def decide(self, t, view) -> Decision:
        return Decision({self.ticker: 1.0})


class EqualWeight(Strategy):
    name = "B1e_equal_weight"
    rebalance = "monthly"

    def decide(self, t, view) -> Decision | None:
        last = view.prices.iloc[-1]
        candidates = set(view.prices.columns[last.notna()])
        if view.members is not None:
            candidates &= set(view.members)
        candidates.discard("SPY")
        if not candidates:
            return None
        w = 1.0 / len(candidates)
        return Decision({tk: w for tk in sorted(candidates)})
