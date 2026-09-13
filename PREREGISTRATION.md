# Pre-Registration

## Design

**Strategies (5):**
- B1: Buy-and-hold SPY (variant B1e: equal-weight point-in-time universe, monthly rebalance)
- B2: Cross-sectional momentum 12-1 (rank by trailing 12-month return skipping most recent month; long top decile equal-weight; monthly rebalance at month-end; minimum 200 valid observations in the ranking window)
- S1: News-sentiment long-short (Lopez-Lira & Tang 2023 prompt, verbatim; firm-day mean score; long score>0 / short score<0, equal-weighted legs, daily rebalance; long-only variant also reported)
- S2: Agentic (FinMem-style layered-memory agent, simplified reference implementation; BUY/HOLD/SELL per ticker per day)
- S3: Naive single-prompt baseline (same inputs as S2, no memory/agent scaffolding)

**Evaluation window (amended by A2):** calendar cutoff 2025-09-01 → 2026-06-30. Because 2025-09-01 was not a trading day, the first evaluation session is 2025-09-02. This is strictly after the documented 2025-08 knowledge cutoff of the pinned `azure/gpt-5.4-mini` deployment; the exact model ID and cutoff are recorded in `config.py` and every result file.

**Universe:** point-in-time S&P 500 membership (two independent reconstruction sources reconciled); LLM strategies run on a fixed random sample of 30 members (seed 42, sampled from membership on 2025-09-01) for cost control. The amended sample is frozen in `data/processed/llm_sample.json` and listed in `data/quality_memo.md`.

**Cost grid:** 0 / 10 / 25 / 50 bps proportional per side, applied to turnover. Turnover reported.

**Metrics:** CAGR, annualized Sharpe (HAC/Newey-West t-stat, lag = floor(1.5 * T^(1/3))), Sortino, max drawdown, Calmar, hit rate, annualized turnover; Deflated Sharpe Ratio with number-of-trials N equal to the count of strategy variants enumerated in this document (N = 8: B1, B1e, B2, S1, S1-long-only, S2, S3, and S3-P1); Sharpe differences vs. B1 and B2 via stationary block bootstrap (mean block 10 days, 10,000 resamples, 95% CI).

**S3 request unit (clarified by A3):** exactly one independent model request per eligible ticker-day. No request combines multiple tickers. Ticker-days without news include an explicit no-news marker.

**S3-P1 prompt arm (specified by A4):** identical to S3 in model, ticker-days, inputs, output schema, temperature, and request unit. Its sole change is replacing the decision sentence with: “Based only on these price statistics and headlines, select BUY, SELL, or HOLD for this stock over the next trading day.” The exact template is `NAIVE_PROMPT_P1` (SHA-256 `08a09ef0afa04880e5aef77ec21f1ec5aa903d2fe516c50cb152092bf4a7b884`), also stored in result metadata.

**Exploratory S3-A robustness arm (added by A5):** identical to S3 over the full primary sample, except explicit occurrences of the focal ticker and a frozen list of the issuer's legal/common names are replaced with `[COMPANY]`. It tests sensitivity to explicit entity identifiers but cannot remove indirect identity clues from products, events, counterparties, or price paths. S3-A is excluded from H1-H3 and the preregistered DSR trial family.

**Rolling windows:** 63-trading-day windows, 21-day step. **Regimes:** bull/bear by SPY vs. 200-day MA; high/low vol by ^VIX median split over the window.

**Ablation (S2 only), 2×2×2:** {in-knowledge-window period (2023-07-01→2024-06-30) ↔ post-cutoff period} × {static current-members ticker set ↔ point-in-time universe} × {gross ↔ net 25 bps}.

## Confirmatory hypotheses

- **H1:** net-of-cost (25 bps) post-cutoff Sharpe(S1) ≤ Sharpe(B2).
- **H2:** ablation contamination main effect > 0 (in-window return > post-cutoff return, all else equal).
- **H3:** |Sharpe(S2) − Sharpe(S3)| indistinguishable from 0 at the 5% level (bootstrap CI covers 0).
