"""Daily price data: Yahoo Finance primary, Stooq cross-check (data-quality standard).

Adjusted closes (splits + dividends) are used for all return computations.
Delisted tickers simply stop having data; the backtest engine treats a missing
return as liquidation-to-cash (documented limitation, see thesis Ch. 3/6).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from ..config import PROCESSED_DIR

CHUNK = 80


def download_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Batch-download adjusted close and volume. Returns (adj_close, volume, missing).

    yfinance's `end` is EXCLUSIVE; we add one day so `end` is included
    (review finding, fixed).
    """
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = sorted(set(tickers))
    closes, volumes = [], []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        df = yf.download(
            chunk, start=start, end=end_excl, auto_adjust=True,
            progress=False, threads=True, group_by="column",
        )
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            closes.append(df["Close"])
            volumes.append(df["Volume"])
        else:  # single-ticker chunk
            closes.append(df[["Close"]].rename(columns={"Close": chunk[0]}))
            volumes.append(df[["Volume"]].rename(columns={"Volume": chunk[0]}))
        time.sleep(0.5)
    adj_close = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(volumes, axis=1).sort_index()
    adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]
    got = set(adj_close.columns[adj_close.notna().any()])
    missing = [t for t in tickers if t not in got]
    adj_close = adj_close.drop(columns=[c for c in adj_close.columns if c not in got])
    volume = volume.reindex(columns=adj_close.columns)
    adj_close.index = pd.to_datetime(adj_close.index).tz_localize(None)
    volume.index = adj_close.index
    return adj_close, volume, missing


def save_prices(adj_close: pd.DataFrame, volume: pd.DataFrame, name: str = "universe") -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adj_close.to_parquet(PROCESSED_DIR / f"{name}_adj_close.parquet")
    volume.to_parquet(PROCESSED_DIR / f"{name}_volume.parquet")


def load_prices(name: str = "universe") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}_adj_close.parquet")


def stooq_cross_check(adj_close: pd.DataFrame, sample: list[str], start: str, end: str) -> pd.DataFrame:
    """Compare Yahoo daily returns with Stooq daily returns on a ticker sample.

    Stooq closes are split-adjusted but not always dividend-adjusted, so small
    ex-date differences are expected; large or frequent disagreement flags a
    data problem. Returns per-ticker report.
    """
    import io as _io

    import requests as _requests

    rows = []
    for t in sample:
        try:
            url = f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"
            resp = _requests.get(url, timeout=30, headers={"User-Agent": "mirage-thesis/0.1"})
            resp.raise_for_status()
            sq = pd.read_csv(_io.StringIO(resp.text), parse_dates=["Date"], index_col="Date").sort_index()
            sq = sq.loc[str(start):str(end)]
        except Exception as e:  # noqa: BLE001 - report and continue
            rows.append({"ticker": t, "status": f"stooq error: {e}"})
            continue
        if t not in adj_close.columns or sq.empty:
            rows.append({"ticker": t, "status": "missing on one side"})
            continue
        a = adj_close[t].pct_change()
        b = sq["Close"].pct_change()
        b.index = pd.to_datetime(b.index).tz_localize(None)
        joined = pd.concat([a, b], axis=1, keys=["yahoo", "stooq"]).dropna()
        if len(joined) < 20:
            rows.append({"ticker": t, "status": "too few overlapping days"})
            continue
        diff = (joined["yahoo"] - joined["stooq"]).abs()
        rows.append({
            "ticker": t,
            "status": "ok",
            "n_days": len(joined),
            "corr": float(joined["yahoo"].corr(joined["stooq"])),
            "median_abs_diff": float(diff.median()),
            "p95_abs_diff": float(diff.quantile(0.95)),
            "share_gt_50bps": float((diff > 0.005).mean()),
        })
        time.sleep(0.3)
    return pd.DataFrame(rows)


def market_consistency_check(adj_close: pd.DataFrame, start: str, end: str) -> dict:
    """Consistency check of the market series across four independent data lines:
    SPY vs IVV vs VOO (three issuers' S&P 500 ETFs) and the ^GSPC index line.

    Near-identical daily returns are expected (tracking noise of a few bps);
    a bad split/dividend adjustment or missing day on any single line shows up
    immediately. This does NOT substitute for a second *vendor* — per-ticker
    validation against Tiingo/CRSP is the university-setting solution (Stooq
    and FRED were unavailable to automated fetch as of 2026-07; documented).
    """
    import yfinance as _yf

    peers = _yf.download(["IVV", "VOO", "^GSPC"], start=start, end=end,
                         auto_adjust=True, progress=False, group_by="column")["Close"]
    peers.index = pd.to_datetime(peers.index).tz_localize(None)
    spy = adj_close["SPY"].pct_change().rename("SPY")
    out = {"base": "Yahoo SPY (adjusted)"}
    for col in peers.columns:
        pr = peers[col].pct_change()
        joined = pd.concat([spy, pr], axis=1).dropna()
        diff = (joined.iloc[:, 0] - joined.iloc[:, 1]).abs()
        out[str(col)] = {
            "n_days": int(len(joined)),
            "corr": float(joined.iloc[:, 0].corr(joined.iloc[:, 1])),
            "median_abs_diff_bps": float(diff.median() * 1e4),
            "p95_abs_diff_bps": float(diff.quantile(0.95) * 1e4),
            "share_gt_50bps": float((diff > 0.005).mean()),
        }
    return out


def to_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; leading NaNs preserved (pre-listing)."""
    return adj_close.pct_change()


def coverage_report(adj_close: pd.DataFrame) -> dict:
    notna = adj_close.notna()
    return {
        "n_tickers": int(adj_close.shape[1]),
        "n_days": int(adj_close.shape[0]),
        "first_date": str(adj_close.index.min().date()),
        "last_date": str(adj_close.index.max().date()),
        "median_coverage": float(notna.mean().median()),
        "tickers_below_50pct_coverage": sorted(notna.mean()[notna.mean() < 0.5].index.tolist()),
        "n_all_nan_days": int((~notna.any(axis=1)).sum()),
    }
