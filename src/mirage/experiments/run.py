"""Run a strategy through the five-standards harness and persist everything."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult, run_backtest
from ..backtest.view import MarketData
from ..config import COST_GRID_BPS, HEADLINE_COST_BPS, PROCESSED_DIR, RESULTS_DIR
from ..data.leakage import assert_no_leakage, audit_decision_log
from ..evaluation import metrics
from ..evaluation.regimes import label_regimes, regime_breakdown
from ..evaluation.rolling import rolling_summary, rolling_windows


def load_market(universe: str = "pit", with_membership=None, news: pd.DataFrame | None = None) -> MarketData:
    from ..data.prices import load_prices

    prices = load_prices()
    membership = with_membership if universe == "pit" else None
    return MarketData(prices, membership=membership, news=news)


def load_benchmark_series() -> tuple[pd.Series, pd.Series]:
    spy = pd.read_parquet(PROCESSED_DIR / "universe_adj_close.parquet")["SPY"]
    vix = pd.read_parquet(PROCESSED_DIR / "vix.parquet")["VIX"]
    return spy, vix


def evaluate(res: BacktestResult, spy: pd.Series, vix: pd.Series) -> dict:
    ret = res.returns
    per_cost = {"gross": metrics.summary(ret["gross"], ret["turnover"])}
    for c in COST_GRID_BPS:
        per_cost[f"net_{c}"] = metrics.summary(ret[f"net_{c}"], ret["turnover"])

    headline = ret[f"net_{HEADLINE_COST_BPS}"]
    regs = label_regimes(spy, vix, ret.index)
    audit_violations = audit_decision_log(res.decisions) if len(res.decisions) else res.decisions

    return {
        "meta": res.meta,
        "per_cost": per_cost,
        "rolling_net_headline": rolling_summary(headline),
        "regimes_net_headline": regime_breakdown(headline, regs),
        "leakage_audit": {
            "n_decisions": int(len(res.decisions)),
            "n_violations": int(len(audit_violations)),
        },
    }


def run_and_save(
    name: str,
    strategy,
    market: MarketData,
    start: str,
    end: str,
    extra_meta: dict | None = None,
    strict_audit: bool = True,
    progress: bool = False,
    progress_label: str | None = None,
) -> dict:
    out = RESULTS_DIR / name
    checkpoint_path = out / "checkpoint.pkl"
    res = run_backtest(
        strategy, market, start, end, COST_GRID_BPS,
        checkpoint_path=checkpoint_path,
        progress=progress,
        progress_label=progress_label or name,
    )
    if strict_audit and len(res.decisions):
        assert_no_leakage(res.decisions)

    spy, vix = load_benchmark_series()
    evals = evaluate(res, spy, vix)
    evals["meta"].update(extra_meta or {})

    out.mkdir(parents=True, exist_ok=True)
    res.returns.to_parquet(out / "returns.parquet")
    if len(res.decisions):
        res.decisions.to_parquet(out / "decisions.parquet")
    if len(res.weights):
        res.weights.to_parquet(out / "weights.parquet")
    rolling_windows(res.returns[f"net_{HEADLINE_COST_BPS}"]).to_csv(out / "rolling_windows.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(evals, indent=2, default=_jsonable))
    checkpoint_path.unlink(missing_ok=True)
    return evals


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, pd.Timestamp):
        return str(x)
    return str(x)


def load_returns(name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / name / "returns.parquet")


def list_results() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.name for p in RESULTS_DIR.iterdir() if (p / "metrics.json").exists())
