"""Shared machinery for S1/S2/S3 runs and the S2 ablation matrix."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from ..backtest.view import MarketData
from ..config import (
    ANTHROPIC_MODEL, EVAL_END, EVAL_START, INWINDOW_END, INWINDOW_START,
    MOCK_MODEL, OPENCODE_MODEL, SECONDARY_MODEL, PROCESSED_DIR,
)
from ..data.news import load_news, save_news, synthetic_news
from ..data.prices import load_prices
from ..data.universe import Membership, download_membership
from ..llm.client import LLMClient
from ..strategies.finmem_agent import FinMemAgent
from ..strategies.naive_prompt import ANONYMIZATION_VERSION, NaivePrompt
from ..strategies.news_sentiment import NewsSentiment
from .run import run_and_save

WINDOWS = {"eval": (EVAL_START, EVAL_END), "inwindow": (INWINDOW_START, INWINDOW_END)}
MODELS = {
    "mock": MOCK_MODEL,
    "anthropic": ANTHROPIC_MODEL,
    "openai": SECONDARY_MODEL,
    "opencode": OPENCODE_MODEL,
}


def llm_sample(window: str = "eval") -> list[str]:
    """Per-window sample: each window's sample is drawn from membership as of that
    window's own start (pre-registration amendment 1 — avoids selecting past-window
    stocks with future membership knowledge)."""
    fname = "llm_sample.json" if window == "eval" else "llm_sample_inwindow.json"
    return json.loads((PROCESSED_DIR / fname).read_text())["tickers"]


def get_news(kind: str, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """kind: 'synthetic' (pipeline validation) or 'alpaca' (real, pre-downloaded)."""
    if kind == "synthetic":
        name = f"synthetic_{start}_{end}"
        try:
            return load_news(name)
        except FileNotFoundError:
            prices = load_prices()
            days = prices.index[(prices.index >= start) & (prices.index <= end)]
            df = synthetic_news(tickers, days)
            save_news(df, name)
            return df
    return load_news(kind)  # real news must be downloaded first (scripts/fetch_news.py)


def build_strategy(key: str, client: LLMClient, tickers: list[str]):
    if key == "s1":
        return NewsSentiment(client, tickers)
    if key == "s1long":
        return NewsSentiment(client, tickers, long_only=True)
    if key == "s2":
        return FinMemAgent(client, tickers)
    if key == "s3":
        return NaivePrompt(client, tickers)
    if key == "s3p1":
        return NaivePrompt(client, tickers, prompt_variant="p1")
    if key == "s3anon":
        return NaivePrompt(client, tickers, anonymized=True)
    raise ValueError(f"unknown strategy {key}")


def run_llm_strategy(strategy_key: str, provider: str, window: str = "eval",
                     universe: str = "pit", news_kind: str = "synthetic",
                     progress: bool = False) -> dict:
    start, end = WINDOWS[window]
    spec = MODELS[provider]
    tickers = llm_sample(window)
    news = get_news(news_kind, tickers, start, end)
    membership = Membership(download_membership()) if universe == "pit" else None
    market = MarketData(load_prices(), membership=membership, news=news)
    client = LLMClient(spec)
    strat = build_strategy(strategy_key, client, tickers)

    prompt_template = getattr(strat, "prompt_template", None)
    strategy_meta = {
        "analysis_role": getattr(strat, "analysis_role", "primary"),
        "llm_request_unit": "ticker_day" if isinstance(strat, (FinMemAgent, NaivePrompt)) else "headline",
    }
    if prompt_template is not None:
        strategy_meta.update({
            "prompt_variant": getattr(strat, "prompt_variant", "original"),
            "prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(),
            "anonymized": getattr(strat, "anonymized", False),
        })
        if getattr(strat, "anonymized", False):
            strategy_meta["anonymization_scope"] = ANONYMIZATION_VERSION

    name = f"{strat.name}__{provider}__{window}__{universe}__{news_kind}"
    evals = run_and_save(
        name, strat, market, start, end,
        extra_meta={
            "provider": spec.provider, "model": spec.model,
            "knowledge_cutoff": spec.knowledge_cutoff,
            "window": window, "universe": universe, "news": news_kind,
            "pipeline_validation_only": news_kind == "synthetic" or provider == "mock",
            **strategy_meta,
        },
        progress=progress,
        progress_label=name,
    )
    # Call counters are only known AFTER the run — update meta and re-persist.
    from ..config import RESULTS_DIR

    evals["meta"]["n_llm_calls"] = client.calls
    evals["meta"]["n_cache_hits"] = client.cache_hits
    metrics_path = RESULTS_DIR / name / "metrics.json"
    metrics_path.write_text(json.dumps(evals, indent=2, default=str))
    return evals


def run_ablation_s2(provider: str, news_kind: str = "synthetic") -> dict:
    """Pre-registered 2x2x2: window x universe x (cost handled by the grid)."""
    cells = {}
    for window in ("inwindow", "eval"):
        for universe in ("static", "pit"):
            evals = run_llm_strategy("s2", provider, window=window, universe=universe, news_kind=news_kind)
            cells[f"{window}|{universe}"] = {
                "gross_total_return": evals["per_cost"]["gross"]["total_return"],
                "gross_sharpe": evals["per_cost"]["gross"]["sharpe"],
                "net25_total_return": evals["per_cost"]["net_25"]["total_return"],
                "net25_sharpe": evals["per_cost"]["net_25"]["sharpe"],
            }
    return cells
