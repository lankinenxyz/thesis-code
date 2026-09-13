"""Common strategy interface: decide(t, view) -> Decision(weights)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Decision:
    weights: dict[str, float] | None       # ticker -> target weight; None = keep drifted
    data_max_ts: pd.Timestamp | None = None  # latest datum used (audit); defaults to view's


class Strategy(ABC):
    name: str = "base"
    rebalance: str = "daily"  # "daily" | "monthly" | "once"

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        _ = state

    @abstractmethod
    def decide(self, t: pd.Timestamp, view) -> Decision | None:
        ...
