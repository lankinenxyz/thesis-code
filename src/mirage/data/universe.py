"""Point-in-time S&P 500 membership (evaluation standard #2).

Primary source: fja05680/sp500 historical-components reconstruction (GitHub).
Reconciliation source: Wikipedia's current constituent list, compared against the
latest snapshot from the primary source. Disagreements are reported, not silently
resolved — see the data-quality memo produced by scripts/build_data.py.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from ..config import RAW_DIR

GITHUB_API = "https://api.github.com/repos/fja05680/sp500/contents/"
RAW_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_ticker(t: str) -> str:
    """Share-class dots (BRK.B) -> dashes (BRK-B), matching Yahoo Finance symbols."""
    return t.strip().upper().replace(".", "-")


def _find_history_csv_name() -> str:
    r = requests.get(GITHUB_API, timeout=30)
    r.raise_for_status()
    names = [f["name"] for f in r.json() if f["name"].lower().endswith(".csv")]
    cands = [n for n in names if "historical components" in n.lower()]
    if not cands:
        raise RuntimeError(f"No historical-components CSV in repo listing: {names}")
    updated = [n for n in cands if "updated" in n.lower()]
    return updated[0] if updated else sorted(cands)[-1]


def download_membership(force: bool = False) -> pd.DataFrame:
    """Return snapshots DataFrame with columns [date, tickers(list[str], normalized)]."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "sp500_membership.csv"
    if not out.exists() or force:
        name = _find_history_csv_name()
        r = requests.get(RAW_BASE + requests.utils.quote(name), timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
    raw = pd.read_csv(io.BytesIO(out.read_bytes()))
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["tickers"] = raw["tickers"].map(
        lambda s: [normalize_ticker(t) for t in str(s).split(",") if t.strip()]
    )
    return raw.sort_values("date").reset_index(drop=True)


class Membership:
    """Point-in-time membership lookup: members(date) uses the latest snapshot <= date."""

    def __init__(self, snapshots: pd.DataFrame):
        snapshots = snapshots.sort_values("date").reset_index(drop=True)
        self._dates = snapshots["date"].to_numpy(dtype="datetime64[ns]")
        self._sets = [frozenset(row) for row in snapshots["tickers"]]

    def members(self, date) -> frozenset:
        d = np.datetime64(pd.Timestamp(date))
        idx = int(np.searchsorted(self._dates, d, side="right")) - 1
        if idx < 0:
            raise ValueError(f"No membership snapshot on or before {date}")
        return self._sets[idx]

    def all_tickers_between(self, start, end) -> set:
        s, e = np.datetime64(pd.Timestamp(start)), np.datetime64(pd.Timestamp(end))
        lo = max(int(np.searchsorted(self._dates, s, side="right")) - 1, 0)
        hi = int(np.searchsorted(self._dates, e, side="right"))
        out: set = set()
        for i in range(lo, hi):
            out |= self._sets[i]
        return out


def current_members_wikipedia() -> set:
    """Current constituents from Wikipedia (reconciliation source)."""
    r = requests.get(WIKI_URL, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    for tbl in tables:
        cols = [str(c).lower() for c in tbl.columns]
        if "symbol" in cols:
            sym = tbl[tbl.columns[cols.index("symbol")]]
            return {normalize_ticker(str(s)) for s in sym.dropna()}
    raise RuntimeError("Wikipedia constituents table not found")


def reconciliation_report(membership: Membership, asof) -> dict:
    """Compare primary source's members(asof) with Wikipedia's current list."""
    primary = set(membership.members(asof))
    wiki = current_members_wikipedia()
    return {
        "asof": str(asof),
        "n_primary": len(primary),
        "n_wikipedia": len(wiki),
        "only_primary": sorted(primary - wiki),
        "only_wikipedia": sorted(wiki - primary),
        "overlap": len(primary & wiki),
    }
