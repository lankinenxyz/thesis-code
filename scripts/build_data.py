"""Phase 1: download membership + prices + VIX, run cross-checks, write quality memo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yfinance as yf

from mirage.config import (
    DATA_DIR, EVAL_END, HISTORY_START, INWINDOW_SAMPLE_DATE, LLM_SAMPLE_DATE,
    LLM_SAMPLE_SEED, LLM_SAMPLE_SIZE, PROCESSED_DIR,
)
from mirage.data.prices import coverage_report, download_prices, market_consistency_check, save_prices
from mirage.data.universe import Membership, download_membership, reconciliation_report


def main() -> None:
    print("1/6 membership snapshots ...")
    snaps = download_membership()
    membership = Membership(snaps)
    tickers = sorted(membership.all_tickers_between(HISTORY_START, EVAL_END))
    print(f"    {len(snaps)} snapshots; {len(tickers)} unique tickers in window")

    print("2/6 prices (Yahoo, chunked) ...")
    cache = PROCESSED_DIR / "universe_adj_close.parquet"
    if cache.exists() and "--force" not in sys.argv:
        adj_close = pd.read_parquet(cache)
        missing = sorted(set(tickers) - set(adj_close.columns))
        print(f"    reusing cached matrix {adj_close.shape}; {len(missing)} tickers absent")
    else:
        adj_close, volume, missing = download_prices(tickers + ["SPY"], HISTORY_START, EVAL_END)
        save_prices(adj_close, volume)
        print(f"    matrix {adj_close.shape}; {len(missing)} tickers with no data (mostly delisted)")

    print("3/6 VIX ...")
    vix_end = (pd.Timestamp(EVAL_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    vix_df = yf.download("^VIX", start=HISTORY_START, end=vix_end, progress=False, auto_adjust=True)
    vix = vix_df["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.rename("VIX").to_frame().to_parquet(PROCESSED_DIR / "vix.parquet")

    print("4/6 market-series consistency check ...")
    xcheck = market_consistency_check(adj_close, HISTORY_START, EVAL_END)

    print("5/6 reconciliation vs Wikipedia ...")
    try:
        recon = reconciliation_report(membership, adj_close.index.max())
    except Exception as e:  # noqa: BLE001
        recon = {"error": str(e)}

    print("6/6 LLM samples + quality memo ...")

    def draw_sample(asof: str, fname: str) -> list[str]:
        pool = sorted(set(membership.members(asof)) & set(adj_close.columns))
        rng = np.random.default_rng(LLM_SAMPLE_SEED)
        picked = sorted(rng.choice(pool, size=LLM_SAMPLE_SIZE, replace=False).tolist())
        (PROCESSED_DIR / fname).write_text(json.dumps(
            {"tickers": picked, "seed": LLM_SAMPLE_SEED, "asof": asof}, indent=2))
        return picked

    llm_sample = draw_sample(LLM_SAMPLE_DATE, "llm_sample.json")
    llm_sample_inwindow = draw_sample(INWINDOW_SAMPLE_DATE, "llm_sample_inwindow.json")

    cov = coverage_report(adj_close)
    memo = ["# Data-Quality Memo (auto-generated)", "",
            f"- Membership snapshots: {len(snaps)}, {snaps['date'].min().date()} to {snaps['date'].max().date()}",
            f"- Unique tickers {HISTORY_START}..{EVAL_END}: {len(tickers)}; no-Yahoo-data (delisted/renamed): {len(missing)}",
            f"  - {missing}",
            f"- Price matrix: {cov['n_days']} days x {cov['n_tickers']} tickers; median coverage {cov['median_coverage']:.1%}",
            f"- LLM sample eval ({LLM_SAMPLE_SIZE}, seed {LLM_SAMPLE_SEED}, as of {LLM_SAMPLE_DATE}): {llm_sample}",
            f"- LLM sample inwindow (as of {INWINDOW_SAMPLE_DATE}): {llm_sample_inwindow}",
            "", "## Survivorship caveat",
            "Tickers with no Yahoo data are predominantly delisted/renamed names. Their absence",
            "biases the *point-in-time* universe slightly toward survivors; documented as a",
            "limitation (thesis Ch. 3 & 6). CRSP would resolve this in a university setting.",
            "", "## Market-series cross-check", json.dumps(xcheck, indent=2),
            "", "## Wikipedia reconciliation", json.dumps(recon, indent=2)[:3000]]
    (DATA_DIR / "quality_memo.md").write_text("\n".join(memo))
    print("done. memo at data/quality_memo.md")


if __name__ == "__main__":
    main()
