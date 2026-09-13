"""Phase 4c: pre-registered 2x2x2 ablation for S2 (window x universe x cost)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mirage.config import RESULTS_DIR
from mirage.experiments.llm_runs import run_ablation_s2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai", "opencode"])
    ap.add_argument("--news", default="synthetic", choices=["synthetic", "alpaca"])
    a = ap.parse_args()

    cells = run_ablation_s2(a.provider, a.news)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"ablation_s2_{a.provider}_{a.news}.json"
    out.write_text(json.dumps(cells, indent=2))
    print(json.dumps(cells, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
