"""Event-driven daily backtest engine (~150 lines, fully auditable).

Conventions (documented in thesis Ch. 4):
- The strategy decides at day t's close using MarketView(t); target weights take
  effect for day t+1's close-to-close return.
- Day t portfolio return uses the weights held coming into t; weights then drift
  by relative returns before any rebalance.
- Transaction cost = cost_bps/1e4 * turnover, charged on the rebalance day,
  where turnover = sum_i |w_target_i - w_drifted_i| (buys + sells, per side).
- A missing return (pre-listing / delisted) is treated as liquidation to cash:
  the position's weight is dropped after registering a 0 return that day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import tempfile
import time

import numpy as np
import pandas as pd

from .view import MarketData, MarketView


@dataclass
class BacktestResult:
    name: str
    returns: pd.DataFrame          # columns: gross, net_<c> per cost, turnover
    decisions: pd.DataFrame        # decision_day, data_max_ts, n_positions, gross_exposure
    weights: pd.DataFrame          # target weights on each rebalance day
    meta: dict = field(default_factory=dict)


def month_end_days(calendar: pd.DatetimeIndex) -> set:
    s = pd.Series(calendar, index=calendar)
    return set(s.groupby([calendar.year, calendar.month]).max())


def run_backtest(
    strategy,
    market: MarketData,
    start,
    end,
    cost_grid=(0, 10, 25, 50),
    checkpoint_path: Path | None = None,
    progress: bool = False,
    progress_label: str | None = None,
) -> BacktestResult:
    cal = market.calendar
    days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    if len(days) == 0:
        raise ValueError("Empty backtest window")
    # Month-ends from the FULL calendar so a truncated final month never fakes
    # a month-end (review finding, fixed).
    me_days = month_end_days(cal)

    weights: dict[str, float] = {}
    gross = pd.Series(0.0, index=days)
    turnover = pd.Series(0.0, index=days)
    dec_rows, weight_rows = [], []
    start_i = 0
    due_indices = [
        i for i, t in enumerate(days)
        if i < len(days) - 1 and (
            strategy.rebalance == "daily"
            or (strategy.rebalance == "monthly" and t in me_days)
            or (strategy.rebalance == "once" and i == 0)
        )
    ]
    total_due = len(due_indices)
    completed_due = 0
    completed_this_session = 0
    progress_start = time.monotonic()

    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt.get("start") == str(pd.Timestamp(start).date()) and ckpt.get("end") == str(pd.Timestamp(end).date()):
            start_i = int(ckpt["next_i"])
            weights = ckpt["weights"]
            gross.update(ckpt["gross"])
            turnover.update(ckpt["turnover"])
            dec_rows = ckpt["dec_rows"]
            weight_rows = ckpt["weight_rows"]
            if hasattr(strategy, "load_state_dict"):
                strategy.load_state_dict(ckpt.get("strategy_state", {}))
            completed_due = sum(1 for j in due_indices if j < start_i)

    if progress:
        label = progress_label or strategy.name
        resume = f" resuming_from_day={start_i + 1}/{len(days)}" if start_i else ""
        print(
            f"progress {label}: {completed_due}/{total_due} decisions complete "
            f"window={days[0].date()}..{days[-1].date()}{resume}",
            flush=True,
        )

    for i in range(start_i, len(days)):
        t = days[i]
        # 1) today's return from weights held coming into t
        r_t = 0.0
        if weights:
            rets = market.returns.loc[t]
            dead = []
            for tk, w in weights.items():
                r_i = rets.get(tk, np.nan)
                if np.isnan(r_i):
                    dead.append(tk)  # liquidated to cash at last price
                else:
                    r_t += w * r_i
            for tk in dead:
                weights.pop(tk)
        gross.loc[t] = r_t

        # 2) drift weights to end-of-day proportions
        if weights and (1.0 + r_t) != 0.0:
            rets = market.returns.loc[t]
            weights = {
                tk: w * (1.0 + (0.0 if np.isnan(rets.get(tk, np.nan)) else rets[tk])) / (1.0 + r_t)
                for tk, w in weights.items()
            }

        # 3) decide at t's close (takes effect tomorrow). No decision on the
        # window's final day: it cannot affect any in-window return and would
        # only charge a phantom trade's costs (review finding, fixed).
        due = i < len(days) - 1 and (
            strategy.rebalance == "daily"
            or (strategy.rebalance == "monthly" and t in me_days)
            or (strategy.rebalance == "once" and i == 0)
        )
        if due:
            prev_day = days[i - 1] if i > 0 else (cal[cal < t][-1] if (cal < t).any() else None)
            view = MarketView(market, t, prev_day)
            if progress:
                print(
                    f"progress {label}: starting {completed_due + 1}/{total_due} "
                    f"date={t.date()}",
                    flush=True,
                )
            decision_start = time.monotonic()
            decision = strategy.decide(t, view)
            if progress:
                completed_due += 1
                completed_this_session += 1
                elapsed = time.monotonic() - progress_start
                avg = elapsed / completed_this_session
                remaining = max(total_due - completed_due, 0)
                client = getattr(strategy, "client", None)
                counters = ""
                if client is not None:
                    counters = f" llm_calls={client.calls} cache_hits={client.cache_hits}"
                print(
                    f"progress {label}: finished {completed_due}/{total_due} "
                    f"date={t.date()} day_elapsed={_fmt_seconds(time.monotonic() - decision_start)} "
                    f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(avg * remaining)}{counters}",
                    flush=True,
                )
            if decision is not None and decision.weights is not None:
                target = {k: float(v) for k, v in decision.weights.items() if v != 0.0}
                tickers = set(target) | set(weights)
                turnover.loc[t] = sum(abs(target.get(k, 0.0) - weights.get(k, 0.0)) for k in tickers)
                weights = target
                dec_rows.append({
                    "decision_day": t,
                    "data_max_ts": decision.data_max_ts or view.data_max_ts,
                    "n_positions": len(target),
                    "gross_exposure": sum(abs(v) for v in target.values()),
                })
                weight_rows.append({"decision_day": t, **target})

        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                {
                    "start": str(pd.Timestamp(start).date()),
                    "end": str(pd.Timestamp(end).date()),
                    "next_i": i + 1,
                    "weights": weights,
                    "gross": gross,
                    "turnover": turnover,
                    "dec_rows": dec_rows,
                    "weight_rows": weight_rows,
                    "strategy_state": strategy.state_dict() if hasattr(strategy, "state_dict") else {},
                },
            )

    returns = pd.DataFrame({"gross": gross, "turnover": turnover})
    for c in cost_grid:
        returns[f"net_{c}"] = gross - turnover * (c / 1e4)

    return BacktestResult(
        name=strategy.name,
        returns=returns,
        decisions=pd.DataFrame(dec_rows),
        weights=pd.DataFrame(weight_rows).set_index("decision_day") if weight_rows else pd.DataFrame(),
        meta={
            "strategy": strategy.name,
            "rebalance": strategy.rebalance,
            "start": str(days[0].date()),
            "end": str(days[-1].date()),
            "n_days": int(len(days)),
            "cost_grid_bps": list(cost_grid),
        },
    )


def _load_checkpoint(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as f:
        pickle.dump(state, f)
        tmp = Path(f.name)
    tmp.replace(path)


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
