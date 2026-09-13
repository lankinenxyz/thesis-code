"""Generate every figure referenced by the thesis from persisted result files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from mirage.config import RESULTS_DIR

FIGURE_DIR = Path(__file__).resolve().parents[2] / "thesis" / "figures"

RUNS = {
    "B1 SPY": "B1_spy_buy_and_hold__2025-09-01__2026-06-30",
    "B2 momentum": "B2_momentum_12_1__2025-09-01__2026-06-30",
    "S1 sentiment": "S1_news_sentiment__opencode__eval__pit__alpaca",
    "S2 FinMem-style": "S2_finmem_agent__opencode__eval__pit__alpaca",
    "S3 naive": "S3_naive_prompt__opencode__eval__pit__alpaca",
}

ROBUSTNESS_RUNS = {
    "S3 canonical": "S3_naive_prompt__opencode__eval__pit__alpaca",
    "S3-P1": "S3_naive_prompt_paraphrase_p1__opencode__eval__pit__alpaca",
    "S3-A exploratory": "S3_naive_prompt_anonymized__opencode__eval__pit__alpaca",
}

COLORS = {
    "B1 SPY": "#374151",
    "B2 momentum": "#0f766e",
    "S1 sentiment": "#b91c1c",
    "S2 FinMem-style": "#c2410c",
    "S3 naive": "#7e22ce",
}


def load_metrics(run: str) -> dict:
    return json.loads((RESULTS_DIR / run / "metrics.json").read_text())


def load_returns(run: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / run / "returns.parquet")


def validate_primary(label: str, run: str, metrics: dict, returns: pd.DataFrame) -> None:
    meta = metrics["meta"]
    assert meta["start"] == "2025-09-02" and meta["end"] == "2026-06-30"
    assert meta["n_days"] == 208 and metrics["leakage_audit"]["n_violations"] == 0
    assert returns.index.is_unique and returns.index.is_monotonic_increasing
    assert len(returns) == 208 and "net_25" in returns
    if label.startswith("S"):
        assert meta["model"] == "azure/gpt-5.4-mini"
        assert meta["knowledge_cutoff"] == "2025-08"
        assert meta["pipeline_validation_only"] is False
    terminal = float((1 + returns["net_25"]).prod() - 1)
    assert np.isclose(terminal, metrics["per_cost"]["net_25"]["total_return"], atol=1e-12)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def headline_performance() -> None:
    metrics = {label: load_metrics(run) for label, run in RUNS.items()}
    returns = {label: load_returns(run) for label, run in RUNS.items()}
    for label, run in RUNS.items():
        validate_primary(label, run, metrics[label], returns[label])
    common_index = returns["B1 SPY"].index
    assert all(frame.index.equals(common_index) for frame in returns.values())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for label, frame in returns.items():
        cumulative = (1 + frame["net_25"]).cumprod() - 1
        width = 2.2 if label in ("B1 SPY", "B2 momentum") else 1.6
        ax.plot(cumulative.index, cumulative, label=label, color=COLORS[label], linewidth=width)
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("A. Cumulative return at 25 bps")
    ax.set_ylabel("Cumulative return")
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    costs = (0, 10, 25, 50)
    keys = ("net_0", "net_10", "net_25", "net_50")
    for label, result in metrics.items():
        values = [result["per_cost"][key]["total_return"] for key in keys]
        ax.plot(costs, values, marker="o", label=label, color=COLORS[label], linewidth=1.8)
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    ax.axvline(25, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.text(25.8, ax.get_ylim()[1] * 0.92, "headline cost", color="#6b7280", fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(costs)
    ax.set_xlabel("Transaction cost (basis points per unit turnover)")
    ax.set_ylabel("Full-window cumulative return")
    ax.set_title("B. Transaction-cost sensitivity")
    fig.tight_layout()
    save(fig, "fig_5_1_headline_performance")


def s2_ablation() -> None:
    cells = json.loads((RESULTS_DIR / "ablation_s2_opencode_alpaca.json").read_text())
    expected = {"inwindow|static", "inwindow|pit", "eval|static", "eval|pit"}
    assert set(cells) == expected

    labels = ("In-window\nstatic", "In-window\nPIT", "Post-cutoff\nstatic", "Post-cutoff\nPIT")
    keys = ("inwindow|static", "inwindow|pit", "eval|static", "eval|pit")
    gross = np.array([cells[key]["gross_total_return"] for key in keys])
    net = np.array([cells[key]["net25_total_return"] for key in keys])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, gross, width, label="Gross", color="#0f766e")
    axes[0].bar(x + width / 2, net, width, label="Net 25 bps", color="#b91c1c")
    axes[0].axhline(0, color="#9ca3af", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Cumulative return")
    axes[0].set_title("A. Complete factorial cells")
    axes[0].legend(frameon=False)

    reported = cells["inwindow|static"]["gross_total_return"]
    post_cutoff = cells["eval|static"]["gross_total_return"]
    pit = cells["eval|pit"]["gross_total_return"]
    compliant = cells["eval|pit"]["net25_total_return"]
    levels = (reported, post_cutoff, pit, compliant)
    waterfall_labels = ("Reported-style", "Post-cutoff", "Point-in-time", "Net 25 bps")
    ax = axes[1]
    ax.bar(0, reported, color="#374151")
    for i, (before, after) in enumerate(zip(levels[:-1], levels[1:]), start=1):
        delta = after - before
        ax.bar(i, abs(delta), bottom=min(before, after), color="#0f766e" if delta >= 0 else "#b91c1c")
        ax.plot([i - 0.55, i - 0.45], [before, before], color="#6b7280", linewidth=0.8)
        ax.text(i, min(before, after) + abs(delta) / 2, f"{delta:+.1%}", ha="center", va="center", color="white", fontsize=8)
    ax.bar(4, compliant, color="#374151")
    ax.text(0, reported + 0.025, f"{reported:+.1%}", ha="center", fontsize=8)
    ax.text(4, compliant - 0.045, f"{compliant:+.1%}", ha="center", fontsize=8)
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    ax.set_xticks(range(5), (waterfall_labels[0], waterfall_labels[1], waterfall_labels[2], waterfall_labels[3], "Compliant"), rotation=15)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Cumulative return")
    ax.set_title("B. Order-specific reported-to-compliant path")
    fig.tight_layout()
    save(fig, "fig_5_2_s2_ablation")


def robustness() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    rolling = []
    for label, run in RUNS.items():
        frame = pd.read_csv(RESULTS_DIR / run / "rolling_windows.csv")
        assert set(("start", "end", "sharpe", "total_return", "max_drawdown")) <= set(frame.columns)
        rolling.append(frame["sharpe"].to_numpy())
    boxes = axes[0].boxplot(rolling, tick_labels=list(RUNS), patch_artist=True, showfliers=False)
    for patch, label in zip(boxes["boxes"], RUNS):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.32)
    rng = np.random.default_rng(0)
    for i, values in enumerate(rolling, start=1):
        axes[0].scatter(i + rng.uniform(-0.07, 0.07, len(values)), values, s=18, color="#374151", zorder=3)
    axes[0].axhline(0, color="#9ca3af", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].set_ylabel("Annualized Sharpe")
    axes[0].set_title("A. Rolling 63-session Sharpe")

    regime_keys = ("trend:bear", "trend:bull", "vol:high_vol", "vol:low_vol")
    regime_labels = ("Bear\n(n=12)", "Bull\n(n=196)", "High VIX\n(n=104)", "Low VIX\n(n=104)")
    offsets = np.linspace(-0.24, 0.24, len(RUNS))
    x = np.arange(len(regime_keys))
    for offset, (label, run) in zip(offsets, RUNS.items()):
        result = load_metrics(run)
        values = [result["regimes_net_headline"][key]["mean_daily_bps"] for key in regime_keys]
        axes[1].scatter(x + offset, values, s=28, label=label, color=COLORS[label])
    axes[1].axhline(0, color="#9ca3af", linewidth=0.8)
    axes[1].set_xticks(x, regime_labels)
    axes[1].set_ylabel("Mean daily return (bps), net 25")
    axes[1].set_title("B. Regime outcomes")
    axes[1].legend(frameon=False, fontsize=7)

    x = np.arange(len(ROBUSTNESS_RUNS))
    gross = []
    net = []
    for run in ROBUSTNESS_RUNS.values():
        result = load_metrics(run)
        gross.append(result["per_cost"]["gross"]["total_return"])
        net.append(result["per_cost"]["net_25"]["total_return"])
    for i, (g, n) in enumerate(zip(gross, net)):
        axes[2].plot([i, i], [g, n], color="#9ca3af", linewidth=1.5)
    axes[2].scatter(x, gross, label="Gross", color="#0f766e", s=42, zorder=3)
    axes[2].scatter(x, net, label="Net 25 bps", color="#b91c1c", s=42, zorder=3)
    axes[2].axhline(0, color="#9ca3af", linewidth=0.8)
    axes[2].set_xticks(x, list(ROBUSTNESS_RUNS), rotation=18)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_ylabel("Full-window cumulative return")
    axes[2].set_title("C. Prompt and identifier sensitivity")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    save(fig, "fig_5_3_robustness")


def main() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    headline_performance()
    s2_ablation()
    robustness()
    print(f"wrote six figure files to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
