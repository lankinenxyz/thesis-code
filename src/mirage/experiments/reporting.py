"""Result-selection helpers shared by reporting scripts and tests."""

from __future__ import annotations

import pandas as pd


def trading_bounds(calendar: pd.DatetimeIndex, start: str, end: str) -> tuple[str, str]:
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if len(days) == 0:
        raise ValueError(f"no trading days in {start}..{end}")
    return str(days[0].date()), str(days[-1].date())


def current_window(meta: dict, bounds: set[tuple[str, str]]) -> bool:
    return (meta.get("start"), meta.get("end")) in bounds


def dsr_trial(meta: dict) -> bool:
    if meta.get("pipeline_validation_only"):
        return False
    if meta.get("window") == "inwindow" or meta.get("universe") == "static":
        return False
    return meta.get("analysis_role", "primary") in {"primary", "preregistered_prompt_arm"}
