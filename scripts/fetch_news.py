"""Fetch real timestamped Alpaca news for the LLM experiment windows.

Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment. Writes
data/processed/news_alpaca.parquet, which is loaded by --news alpaca runs.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mirage.config import EVAL_END, EVAL_START, INWINDOW_END, INWINDOW_START, PROCESSED_DIR
from mirage.data.news import fetch_alpaca_news, save_news


def _chunks(xs: list[str], n: int) -> list[list[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def _sample_tickers() -> list[str]:
    eval_tickers = json.loads((PROCESSED_DIR / "llm_sample.json").read_text())["tickers"]
    inwindow_tickers = json.loads((PROCESSED_DIR / "llm_sample_inwindow.json").read_text())["tickers"]
    return sorted(set(eval_tickers) | set(inwindow_tickers))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds between Alpaca requests")
    ap.add_argument("--name", default="alpaca", help="Output name; default writes news_alpaca.parquet")
    args = ap.parse_args()

    tickers = _sample_tickers()
    windows = [(INWINDOW_START, INWINDOW_END), (EVAL_START, EVAL_END)]
    frames: list[pd.DataFrame] = []
    api_key = os.environ.get("ALPACA_API_KEY") or getpass.getpass("ALPACA_API_KEY: ")
    api_secret = os.environ.get("ALPACA_SECRET_KEY") or getpass.getpass("ALPACA_SECRET_KEY: ")

    print(f"fetching Alpaca news for {len(tickers)} unique LLM-sample tickers")
    for start, end in windows:
        for batch in _chunks(tickers, args.chunk_size):
            print(f"  {start}..{end}: {','.join(batch)}")
            df = fetch_alpaca_news(
                batch,
                f"{start}T00:00:00Z",
                f"{end}T23:59:59Z",
                api_key=api_key,
                api_secret=api_secret,
            )
            if len(df):
                frames.append(df)
            time.sleep(args.sleep)

    if not frames:
        raise RuntimeError(
            "No Alpaca news returned. Check API credentials, historical-news access, "
            "date range, and ticker coverage."
        )

    news = pd.concat(frames, ignore_index=True)
    keep = pd.Series(False, index=news.index)
    for start, end in windows:
        lo = pd.Timestamp(f"{start}T00:00:00Z")
        hi = pd.Timestamp(f"{end}T23:59:59Z")
        keep |= (news["timestamp_utc"] >= lo) & (news["timestamp_utc"] <= hi)
    news = news.loc[keep & news["ticker"].isin(tickers)]
    news = news.drop_duplicates(subset=["id", "ticker"]).sort_values("timestamp_utc").reset_index(drop=True)
    save_news(news, args.name)

    out = PROCESSED_DIR / f"news_{args.name}.parquet"
    print(f"saved {len(news):,} rows to {out}")
    print(news.groupby(news["timestamp_utc"].dt.year).size().to_string())


if __name__ == "__main__":
    main()
