"""S1: news-sentiment long-short, following Lopez-Lira & Tang (2023).

Each headline in (previous close, today's close] is scored by the LLM with their
prompt (verbatim below); scores map YES=+1, UNKNOWN=0, NO=-1; the firm-day score
is the mean over its headlines. Long every firm with score>0 (equal-weighted leg
summing to +1), short every firm with score<0 (leg summing to -1). Daily rebalance.
A long-only variant keeps only the long leg (sums to +1).
"""

from __future__ import annotations

import pandas as pd

from .base import Decision, Strategy

LLT_PROMPT = (
    "Forget all your previous instructions. Pretend you are a financial expert. "
    "You are a financial expert with stock recommendation experience. "
    'Answer "YES" if good news, "NO" if bad news, or "UNKNOWN" if uncertain in the '
    "first line. Then elaborate with one short and concise sentence on the next line. "
    "Is this headline good or bad for the stock price of {company} in the short term?\n"
    "Headline: {headline}"
)


def parse_llt(response: str) -> int:
    first = next((ln.strip().upper().strip('".,!') for ln in response.splitlines() if ln.strip()), "")
    if first.startswith("YES"):
        return 1
    if first.startswith("NO"):
        return -1
    return 0


class NewsSentiment(Strategy):
    rebalance = "daily"

    def __init__(self, client, tickers: list[str], long_only: bool = False):
        self.client = client
        self.tickers = sorted(set(tickers))
        self.long_only = long_only
        self.name = "S1_news_sentiment" + ("_long_only" if long_only else "")

    def decide(self, t, view) -> Decision:
        active = set(self.tickers)
        if view.members is not None:
            active &= set(view.members)
        news = view.news_window(tickers=active)

        scores: dict[str, float] = {}
        max_ts = None
        for tk, group in news.groupby("ticker"):
            vals = []
            for _, row in group.iterrows():
                resp = self.client.complete(
                    LLT_PROMPT.format(company=tk, headline=row["headline"])
                )
                vals.append(parse_llt(resp))
                ts = row["timestamp_utc"]
                max_ts = ts if max_ts is None or ts > max_ts else max_ts
            scores[tk] = sum(vals) / len(vals)

        longs = sorted(tk for tk, s in scores.items() if s > 0)
        shorts = sorted(tk for tk, s in scores.items() if s < 0)
        weights: dict[str, float] = {}
        if longs:
            for tk in longs:
                weights[tk] = 1.0 / len(longs)
        if shorts and not self.long_only:
            for tk in shorts:
                weights[tk] = -1.0 / len(shorts)
        # No signal today -> flat (all cash), which is itself a decision.
        return Decision(weights, data_max_ts=max_ts)
