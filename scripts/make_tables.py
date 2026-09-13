"""Phase 4d: aggregate all results into thesis-ready markdown tables (results/TABLES.md).

Includes: headline metrics x cost grid, rolling-window distributions, regime
breakdown, DSR across pre-registered trials, and bootstrap Sharpe-difference
tests vs B1 and B2 at the headline cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mirage.config import (
    BOOTSTRAP_BLOCK, BOOTSTRAP_RESAMPLES, DSR_NUM_TRIALS, EVAL_END, EVAL_START,
    HEADLINE_COST_BPS, INWINDOW_END, INWINDOW_START, RESULTS_DIR,
)
from mirage.evaluation.bootstrap import sharpe_diff_test
from mirage.evaluation.dsr import daily_sharpe, deflated_sharpe
from mirage.data.prices import load_prices
from mirage.experiments.run import list_results, load_returns
from mirage.experiments.reporting import current_window, dsr_trial, trading_bounds

HEAD = f"net_{HEADLINE_COST_BPS}"

PRIMARY_RUNS = {
    "B1 SPY": "B1_spy_buy_and_hold__2025-09-01__2026-06-30",
    "B1e equal weight": "B1e_equal_weight__2025-09-01__2026-06-30",
    "B2 momentum": "B2_momentum_12_1__2025-09-01__2026-06-30",
    "S1 sentiment": "S1_news_sentiment__opencode__eval__pit__alpaca",
    "S1 long-only": "S1_news_sentiment_long_only__opencode__eval__pit__alpaca",
    "S2 FinMem-style": "S2_finmem_agent__opencode__eval__pit__alpaca",
    "S3 naive": "S3_naive_prompt__opencode__eval__pit__alpaca",
    "S3-P1": "S3_naive_prompt_paraphrase_p1__opencode__eval__pit__alpaca",
    "S3-A": "S3_naive_prompt_anonymized__opencode__eval__pit__alpaca",
}


def fmt(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:+.2%}" if pct else f"{x:.{nd}f}"


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    names = list_results()
    if not names:
        print("no results found; run experiments first")
        return
    metas = {n: json.loads((RESULTS_DIR / n / "metrics.json").read_text()) for n in names}
    calendar = load_prices().index
    bounds = {
        trading_bounds(calendar, EVAL_START, EVAL_END),
        trading_bounds(calendar, INWINDOW_START, INWINDOW_END),
    }
    names = [n for n in names if current_window(metas[n]["meta"], bounds)]
    if not names:
        print("no results found for the currently configured trading windows")
        return
    rets = {n: load_returns(n) for n in names}
    research = [n for n in names if not metas[n]["meta"].get("pipeline_validation_only")]
    missing = set(PRIMARY_RUNS.values()) - set(research)
    if missing:
        raise RuntimeError(f"missing primary results: {sorted(missing)}")

    lines = ["# Results Tables (auto-generated — do not hand-edit)", ""]

    # --- Table 1: headline metrics at each cost ------------------------------
    lines += ["## Table 1 — Performance by strategy and transaction cost", "",
              "| Run | Cost | Total ret | CAGR | Sharpe | NW t | Sortino | MaxDD | Ann. turnover | Validation-only |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for n in research:
        m = metas[n]
        vo = "YES" if m["meta"].get("pipeline_validation_only") else "no"
        for cost_key in ("gross", "net_10", "net_25", "net_50"):
            pc = m["per_cost"].get(cost_key)
            if not pc:
                continue
            lines.append(
                f"| {n} | {cost_key} | {fmt(pc['total_return'], pct=True)} | {fmt(pc['cagr'], pct=True)} "
                f"| {fmt(pc['sharpe'])} | {fmt(pc['nw_tstat'])} | {fmt(pc['sortino'])} "
                f"| {fmt(pc['max_drawdown'], pct=True)} | {fmt(pc.get('ann_turnover'))} | {vo} |")
    lines.append("")

    # --- Table 2: rolling windows --------------------------------------------
    lines += [f"## Table 2 — Rolling 63-day windows ({HEAD})", "",
              "| Run | n windows | Sharpe median [q25, q75] | Sharpe min/max | % windows positive |",
              "|---|---|---|---|---|"]
    for n in research:
        r = metas[n].get("rolling_net_headline", {})
        if not r or r.get("n_windows", 0) == 0:
            continue
        lines.append(
            f"| {n} | {r['n_windows']} | {fmt(r['sharpe_median'])} [{fmt(r['sharpe_q25'])}, {fmt(r['sharpe_q75'])}] "
            f"| {fmt(r['sharpe_min'])} / {fmt(r['sharpe_max'])} | {r['share_windows_positive']:.0%} |")
    lines.append("")

    # --- Table 3: regimes ------------------------------------------------------
    lines += [f"## Table 3 — Regime breakdown ({HEAD})", "",
              "| Run | Regime | n days | Sharpe | Mean daily (bps) | Total ret |",
              "|---|---|---|---|---|---|"]
    for n in research:
        for reg, v in metas[n].get("regimes_net_headline", {}).items():
            lines.append(f"| {n} | {reg} | {v['n_days']} | {fmt(v['sharpe'])} "
                         f"| {fmt(v['mean_daily_bps'])} | {fmt(v['total_return'], pct=True)} |")
    lines.append("")

    # --- Table 4: DSR across pre-registered trials -----------------------------
    # Trial set = REAL runs only. Pipeline-validation runs (mock/synthetic) are
    # harness tests, not research trials; including them inflates trial variance
    # and hence SR0, making DSR uninformative for everyone.
    real = [n for n in names if dsr_trial(metas[n]["meta"])]
    trial_srs = [daily_sharpe(rets[n][HEAD]) for n in real]
    trial_srs = [s for s in trial_srs if s == s]
    n_trials = max(DSR_NUM_TRIALS, len(trial_srs))
    lines += [f"## Table 4 — Deflated Sharpe Ratio (N trials = {n_trials}; "
              f"trial variance from {len(trial_srs)} real runs)", "",
              "| Run | daily SR | SR0 (exp. max unskilled) | DSR = P[skill] |", "|---|---|---|---|"]
    for n in real:
        d = deflated_sharpe(rets[n][HEAD], trial_srs, n_trials=n_trials)
        lines.append(f"| {n} | {fmt(d.get('sr_hat_daily'), nd=3)} | {fmt(d.get('sr0_daily'), nd=3)} | {fmt(d.get('dsr'), nd=3)} |")
    lines.append("")

    # --- Table 5: bootstrap Sharpe differences vs baselines --------------------
    # Benchmarks pinned to the EVAL window; cross-window comparisons are
    # meaningless (review finding, fixed). Non-overlapping runs drop out via
    # the inner join inside sharpe_diff_test.
    b1 = next((n for n in names if n.startswith("B1_spy")), None)
    b2 = next((n for n in names if n.startswith("B2_")), None)
    lines += [f"## Table 5 — Stationary-bootstrap Sharpe differences ({HEAD}, "
              f"{BOOTSTRAP_RESAMPLES} resamples, block {BOOTSTRAP_BLOCK}d)", "",
              "| Run | vs | ann. ΔSharpe | 95% CI (ann.) | p |", "|---|---|---|---|---|"]
    comparisons = [
        n for n in research
        if metas[n]["meta"].get("window") != "inwindow"
        and metas[n]["meta"].get("universe") != "static"
    ]
    for n in comparisons:
        for bench, bname in ((b1, "B1 SPY"), (b2, "B2 momentum")):
            if bench is None or n == bench:
                continue
            t = sharpe_diff_test(rets[n][HEAD], rets[bench][HEAD],
                                 n_resamples=BOOTSTRAP_RESAMPLES, mean_block=BOOTSTRAP_BLOCK)
            if t["diff_daily"] != t["diff_daily"]:
                continue
            lines.append(f"| {n} | {bname} | {fmt(t['diff_annualized'])} "
                         f"| [{fmt(t['ci_lo_annualized'])}, {fmt(t['ci_hi_annualized'])}] | {fmt_p(t['p_value'])} |")
    lines.append("")

    # --- Table 6: direct confirmatory and robustness contrasts -----------------
    direct = (
        ("H1", "S1 sentiment", "B2 momentum", "confirmatory"),
        ("H3", "S2 FinMem-style", "S3 naive", "confirmatory"),
        ("Prompt", "S3-P1", "S3 naive", "preregistered robustness"),
        ("Masking", "S3-A", "S3 naive", "exploratory"),
    )
    lines += ["## Table 6 — Direct paired Sharpe contrasts (net_25)", "",
              "| Test | A − B | Role | ann. ΔSharpe | 95% CI (ann.) | p |",
              "|---|---|---|---|---|---|"]
    for test, a, b, role in direct:
        out = sharpe_diff_test(
            rets[PRIMARY_RUNS[a]][HEAD], rets[PRIMARY_RUNS[b]][HEAD],
            n_resamples=BOOTSTRAP_RESAMPLES, mean_block=BOOTSTRAP_BLOCK,
        )
        lines.append(
            f"| {test} | {a} − {b} | {role} | {fmt(out['diff_annualized'])} "
            f"| [{fmt(out['ci_lo_annualized'])}, {fmt(out['ci_hi_annualized'])}] "
            f"| {fmt_p(out['p_value'])} |"
        )
    lines.append("")

    # --- Table 7: S2 factorial cells and marginal effects ----------------------
    ablation = json.loads((RESULTS_DIR / "ablation_s2_opencode_alpaca.json").read_text())
    lines += ["## Table 7 — S2 2×2×2 ablation", "",
              "| Window | Universe | Gross return | Gross Sharpe | Net-25 return | Net-25 Sharpe |",
              "|---|---|---|---|---|---|"]
    for window, window_label in (("inwindow", "In-knowledge"), ("eval", "Post-cutoff")):
        for universe, universe_label in (("static", "Static"), ("pit", "Point-in-time")):
            cell = ablation[f"{window}|{universe}"]
            lines.append(
                f"| {window_label} | {universe_label} "
                f"| {fmt(cell['gross_total_return'], pct=True)} | {fmt(cell['gross_sharpe'])} "
                f"| {fmt(cell['net25_total_return'], pct=True)} | {fmt(cell['net25_sharpe'])} |"
            )
    lines.append("")

    def marginal(metric: str, factor: str) -> float:
        if factor == "window":
            left = [ablation[f"inwindow|{u}"][metric] for u in ("static", "pit")]
            right = [ablation[f"eval|{u}"][metric] for u in ("static", "pit")]
        else:
            left = [ablation[f"{w}|pit"][metric] for w in ("inwindow", "eval")]
            right = [ablation[f"{w}|static"][metric] for w in ("inwindow", "eval")]
        return sum(left) / len(left) - sum(right) / len(right)

    gross_window_return = marginal("gross_total_return", "window")
    net_window_return = marginal("net25_total_return", "window")
    gross_window_sharpe = marginal("gross_sharpe", "window")
    net_window_sharpe = marginal("net25_sharpe", "window")
    averaged_window_return = (gross_window_return + net_window_return) / 2
    averaged_window_sharpe = (gross_window_sharpe + net_window_sharpe) / 2
    cost_return = sum(c["net25_total_return"] - c["gross_total_return"] for c in ablation.values()) / 4
    cost_sharpe = sum(c["net25_sharpe"] - c["gross_sharpe"] for c in ablation.values()) / 4
    lines += ["### Marginal effects", "",
              "| Effect | Return effect | Sharpe effect |",
              "|---|---|---|",
              f"| In-knowledge − post-cutoff, gross | {fmt(gross_window_return, pct=True)} | {fmt(gross_window_sharpe)} |",
              f"| In-knowledge − post-cutoff, net-25 | {fmt(net_window_return, pct=True)} | {fmt(net_window_sharpe)} |",
              f"| In-knowledge − post-cutoff, averaged over cost | {fmt(averaged_window_return, pct=True)} | {fmt(averaged_window_sharpe)} |",
              f"| Point-in-time − static, averaged | {fmt(marginal('gross_total_return', 'universe'), pct=True)} | {fmt(marginal('gross_sharpe', 'universe'))} |",
              f"| Net-25 − gross, averaged | {fmt(cost_return, pct=True)} | {fmt(cost_sharpe)} |", ""]

    # --- Table 8: primary audit and execution-local call counters --------------
    lines += ["## Table 8 — Primary leakage audits and execution counters", "",
              "| Run | Decisions | Violations | Backend calls | Cache hits |",
              "|---|---|---|---|---|"]
    for label, n in PRIMARY_RUNS.items():
        m = metas[n]
        lines.append(
            f"| {label} | {m['leakage_audit']['n_decisions']} | {m['leakage_audit']['n_violations']} "
            f"| {m['meta'].get('n_llm_calls', '—')} | {m['meta'].get('n_cache_hits', '—')} |"
        )
    lines += ["", "Counters are execution-local. Cache-populating runs and resumed runs therefore cannot be summed as experiment-wide request totals.", ""]

    out = RESULTS_DIR / "TABLES.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(research)} research runs)")


if __name__ == "__main__":
    main()
