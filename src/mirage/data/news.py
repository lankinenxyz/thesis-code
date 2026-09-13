"""News data: Alpaca News API loader (real runs) + deterministic synthetic generator
(pipeline validation ONLY — synthetic runs are never reported as findings).

Schema (parquet): id, timestamp_utc (tz-aware UTC), ticker, headline, source.
One row per (article, ticker) pair.
"""

from __future__ import annotations

import hashlib
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from ..config import PROCESSED_DIR

ALPACA_URL = "https://data.alpaca.markets/v1beta1/news"
ET = ZoneInfo("America/New_York")

# NYSE 13:00-ET early closes, 2022-2026, per the published NYSE holiday calendar
# (July-3-type, day-after-Thanksgiving, and Christmas-Eve half days). On these
# days the news cutoff must be the ACTUAL close, or afternoon headlines leak
# into a trade priced at 13:00 (review finding, fixed).
EARLY_CLOSES = {
    "2022-11-25",
    "2023-07-03", "2023-11-24",
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
}


def close_utc(day) -> pd.Timestamp:
    """UTC timestamp of the NYSE close (16:00 ET; 13:00 ET on early-close days)."""
    d = pd.Timestamp(day)
    hour = 13 if d.strftime("%Y-%m-%d") in EARLY_CLOSES else 16
    return pd.Timestamp(d.year, d.month, d.day, hour, 0, tz=ET).tz_convert("UTC")


# --------------------------------------------------------------------------- #
# Real source
# --------------------------------------------------------------------------- #

def fetch_alpaca_news(tickers: list[str], start: str, end: str,
                      api_key: str | None = None, api_secret: str | None = None) -> pd.DataFrame:
    """Fetch timestamped headlines from Alpaca (free tier). Requires API keys."""
    key = api_key or os.environ.get("ALPACA_API_KEY")
    secret = api_secret or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY / ALPACA_SECRET_KEY for real news download")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    rows, token = [], None
    while True:
        params = {
            "symbols": ",".join(tickers), "start": start, "end": end,
            "limit": 50, "include_content": "false", "sort": "asc",
        }
        if token:
            params["page_token"] = token
        r = requests.get(ALPACA_URL, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        for a in payload.get("news", []):
            for sym in a.get("symbols", []):
                rows.append({
                    "id": str(a["id"]),
                    "timestamp_utc": pd.Timestamp(a["created_at"]).tz_convert("UTC"),
                    "ticker": sym.upper(),
                    "headline": a["headline"],
                    "source": a.get("source", "alpaca"),
                })
        token = payload.get("next_page_token")
        if not token:
            break
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Synthetic source (pipeline validation only)
# --------------------------------------------------------------------------- #

_POS = [
    "{t} beats quarterly earnings estimates, raises full-year guidance",
    "{t} announces record revenue and expanded share buyback",
    "Analysts upgrade {t} to buy citing strong demand",
    "{t} wins major new contract, shares seen higher",
]
_NEG = [
    "{t} misses earnings expectations, cuts outlook",
    "{t} faces regulatory probe over accounting practices",
    "Analysts downgrade {t} on weakening margins",
    "{t} announces product recall; costs expected to rise",
]
_NEU = [
    "{t} to present at industry conference next week",
    "{t} appoints new member to board of directors",
    "{t} announces date of next earnings call",
]


def synthetic_news(tickers: list[str], trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic synthetic headlines with a hidden ground-truth label.

    Seeded by (ticker, date) so every run is identical. `synthetic_label` in
    {+1, 0, -1} exists so unit tests can verify the S1 scoring path end-to-end.
    """
    rows = []
    for t in tickers:
        for day in trading_days:
            seed = int(hashlib.md5(f"{t}|{day.date()}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            n = int(rng.integers(0, 3))  # 0..2 headlines per stock-day
            for k in range(n):
                u = rng.random()
                if u < 0.35:
                    tmpl, label = _POS[int(rng.integers(len(_POS)))], 1
                elif u < 0.70:
                    tmpl, label = _NEG[int(rng.integers(len(_NEG)))], -1
                else:
                    tmpl, label = _NEU[int(rng.integers(len(_NEU)))], 0
                hour = int(rng.integers(9, 16))
                ts = pd.Timestamp(day.year, day.month, day.day, hour, 15, tz=ET).tz_convert("UTC")
                rows.append({
                    "id": f"syn-{t}-{day.date()}-{k}",
                    "timestamp_utc": ts,
                    "ticker": t,
                    "headline": tmpl.format(t=t),
                    "source": "synthetic",
                    "synthetic_label": label,
                })
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def save_news(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"news_{name}.parquet")


def load_news(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"news_{name}.parquet")
