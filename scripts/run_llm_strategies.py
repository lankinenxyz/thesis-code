"""Phase 4b: run S1/S2/S3.

Examples:
  # pipeline validation (no API cost, synthetic news, deterministic mock LLM):
  python scripts/run_llm_strategies.py --strategy s1 --provider mock
  # real run (requires opencode auth and pre-downloaded Alpaca news):
  python scripts/run_llm_strategies.py --strategy s1 --provider opencode --news alpaca
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mirage.experiments.llm_runs import run_llm_strategy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy", required=True,
        choices=["s1", "s1long", "s2", "s3", "s3p1", "s3anon"],
    )
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai", "opencode"])
    ap.add_argument("--window", default="eval", choices=["eval", "inwindow"])
    ap.add_argument("--universe", default="pit", choices=["pit", "static"])
    ap.add_argument("--news", default="synthetic", choices=["synthetic", "alpaca"])
    ap.add_argument("--no-progress", action="store_true", help="Disable decision-day progress output")
    a = ap.parse_args()

    if a.provider == "mock" or a.news == "synthetic":
        print("NOTE: mock/synthetic run — pipeline validation only, never a finding.")
    evals = run_llm_strategy(
        a.strategy, a.provider, a.window, a.universe, a.news,
        progress=not a.no_progress,
    )
    head = evals["per_cost"]["net_25"]
    print(f"done: net25 total={head['total_return']:+.2%} sharpe={head['sharpe']:.2f} "
          f"llm_calls={evals['meta']['n_llm_calls']} cache_hits={evals['meta']['n_cache_hits']}")


if __name__ == "__main__":
    main()
