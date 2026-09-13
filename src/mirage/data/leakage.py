"""Leakage audit (evaluation standard #1 at the data layer).

Every backtest decision is logged as (decision_day, data_max_ts): the latest
timestamp of any datum the strategy touched. The audit asserts that no decision
used data after that day's close. Run on every experiment; CI-fails on violation.
"""

from __future__ import annotations

import pandas as pd

from .news import close_utc


def audit_decision_log(log: pd.DataFrame) -> pd.DataFrame:
    """Return the violating rows (empty DataFrame == clean audit).

    Expects columns: decision_day (naive date-like), data_max_ts (tz-aware UTC
    or naive trading-day timestamp).
    """
    if log.empty:
        return log
    log = log.copy()
    cutoffs = log["decision_day"].map(close_utc)
    ts = log["data_max_ts"].map(_to_utc)
    return log[ts > cutoffs]


def _to_utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        # Naive timestamps are trading-day dates: interpret as that day's close.
        return close_utc(t)
    return t.tz_convert("UTC")


def assert_no_leakage(log: pd.DataFrame) -> None:
    bad = audit_decision_log(log)
    if len(bad):
        raise AssertionError(
            f"LEAKAGE: {len(bad)} decisions used data after the decision-day close:\n"
            f"{bad.head(10)}"
        )
