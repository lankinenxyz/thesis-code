"""Phase 4a: run B1, B1e, B2 over the pre-registered evaluation window."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mirage.backtest.view import MarketData
from mirage.config import EVAL_END, EVAL_START
from mirage.data.prices import load_prices
from mirage.data.universe import Membership, download_membership
from mirage.experiments.run import run_and_save
from mirage.strategies.buy_and_hold import BuyAndHold, EqualWeight
from mirage.strategies.momentum import Momentum


def main(start: str = EVAL_START, end: str = EVAL_END) -> None:
    membership = Membership(download_membership())
    market = MarketData(load_prices(), membership=membership)
    for strat in (BuyAndHold(), EqualWeight(), Momentum()):
        name = f"{strat.name}__{start}__{end}"
        evals = run_and_save(name, strat, market, start, end,
                             extra_meta={"universe": "pit", "window": f"{start}..{end}"})
        head = evals["per_cost"]["net_25"]
        print(f"{strat.name:28s} net25 total={head['total_return']:+.2%} "
              f"sharpe={head['sharpe']:.2f} mdd={head['max_drawdown']:.2%}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()
