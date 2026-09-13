# Thesis Code

Evaluation harness for the master's thesis *"A Contamination-Controlled,
Cost-Inclusive Evaluation of LLM-Based Equity Trading Strategies."*

## What this is

A test-covered Python package that evaluates LLM trading strategies under
five minimum standards simultaneously: contamination control (frozen snapshots, strictly
post-knowledge-cutoff data), point-in-time S&P 500 universe, rolling-window reporting,
net-of-cost returns (0/10/25/50 bps grid with turnover accounting), and regime coverage.
Strategies see data only through a `MarketView` that structurally cannot expose
post-decision data; every decision is leakage-audited; every LLM response is cached in
SQLite so completed runs replay bit-identically without API access.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                              # 33 tests
.venv/bin/python scripts/build_data.py        # free data: membership, prices, VIX (~5 min)
.venv/bin/python scripts/run_baselines.py     # B1 SPY, B1e equal-weight, B2 momentum (real results)
# pipeline validation (no API cost; deterministic mock + synthetic news):
.venv/bin/python scripts/run_llm_strategies.py --strategy s1 --provider mock
.venv/bin/python scripts/run_ablation.py --provider mock
.venv/bin/python scripts/make_tables.py       # results/TABLES.md
.venv/bin/python scripts/make_figures.py      # ../thesis/figures/*.{png,pdf}
```

## Completed real LLM runs

Primary runs use `azure/gpt-5.4-mini` (documented cutoff 2025-08) from the
2025-09-01 calendar cutoff; the first trading session is 2025-09-02. Final
tables are in `results/TABLES.md`, the S2 matrix is in
`results/ablation_s2_opencode_alpaca.json`, and thesis figures are generated
under `../thesis/figures/`.

```bash
opencode models | grep 'azure/gpt-5.4-mini'   # confirm the configured model is available
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...   # free tier, for timestamped news
.venv/bin/pip install -e ".[dev]"
# 1) download news for the sample and windows
.venv/bin/python scripts/fetch_news.py
# 2) replay completed runs from cache, or regenerate with provider access:
.venv/bin/python scripts/run_llm_strategies.py --strategy s1 --provider opencode --news alpaca
.venv/bin/python scripts/run_llm_strategies.py --strategy s1long --provider opencode --news alpaca
.venv/bin/python scripts/run_llm_strategies.py --strategy s2 --provider opencode --news alpaca
.venv/bin/python scripts/run_llm_strategies.py --strategy s3 --provider opencode --news alpaca
.venv/bin/python scripts/run_llm_strategies.py --strategy s3p1 --provider opencode --news alpaca
.venv/bin/python scripts/run_llm_strategies.py --strategy s3anon --provider opencode --news alpaca
.venv/bin/python scripts/run_ablation.py --provider opencode --news alpaca
.venv/bin/python scripts/make_tables.py
.venv/bin/python scripts/make_figures.py
```

The opencode provider shells out to `opencode run --model azure/gpt-5.4-mini` and still
uses the SQLite prompt cache. Primary result metadata must record exactly this model and
the 2025-08 cutoff; results from another model are non-primary.

LLM backtests checkpoint after each completed trading day to
`results/<run>/checkpoint.pkl`. If a run is interrupted, rerun the same command and
it resumes from the checkpoint; completed runs remove their checkpoint after writing
the final result files.

Mock-provider / synthetic-news outputs are flagged `pipeline_validation_only` in every
results file and are never findings.

## Layout

- `src/mirage/data/` — universe (point-in-time membership), prices (+consistency checks), news, leakage audit
- `src/mirage/llm/` — provider-agnostic client (opencode/anthropic/openai/mock), SQLite prompt cache
- `src/mirage/strategies/` — B1/B1e, B2 momentum 12-1, S1 López-Lira–Tang sentiment, S2 FinMem-style agent, S3 naive prompt
- `src/mirage/backtest/` — MarketView (no-leakage-by-construction) + ~150-line engine
- `src/mirage/evaluation/` — metrics (HAC t), Deflated Sharpe, stationary bootstrap, rolling windows, regimes
- `src/mirage/experiments/` + `scripts/` — orchestration, ablation, table generation
