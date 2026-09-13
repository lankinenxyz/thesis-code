"""S2: simplified FinMem-style agent (Yu et al. 2023, arXiv:2311.13743).

Faithful-in-spirit reference implementation with the three FinMem ingredients:
  1. Profiling  - a fixed trader persona in the prompt;
  2. Layered memory - short (7d, keep 3) / mid (30d, keep 2) / long (90d, keep 1)
     stores of headline digests and outcome reflections, retrieved by recency;
  3. Decision module - one LLM call per (ticker, day) returning BUY/SELL/HOLD
     with confidence; weights are confidence-signed and gross-normalized to <= 1.

Divergences from the original (documented in thesis Ch. 4): no embedding-based
retrieval (recency only), no importance scores, single persona. For the real
thesis runs, swap in the original FinMem repo or the FINSABER wrapper via this
same Strategy interface.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .base import Decision, Strategy

PROFILE = (
    "an experienced equity trader with moderate risk appetite who weighs news "
    "flow against price momentum and admits uncertainty when evidence is mixed"
)

ACTION_PROMPT = """You are {profile}.
Today is {date}. You are deciding your position in {ticker} for the next trading day.

Recent price action:
- 5-day return: {r5:.2f}%
- 20-day return: {r20:.2f}%

Your memory of recent events and your own past reflections:
{memories}

Today's headlines for {ticker}:
{headlines}

Respond with a single JSON object on one line:
{{"action": "BUY" | "SELL" | "HOLD", "confidence": <number 0..1>, "reason": "<one short sentence>"}}"""

_LAYERS = ((7, 3), (30, 2), (90, 1))  # (horizon_days, keep_k) short/mid/long


def parse_action(response: str) -> tuple[str, float]:
    try:
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            action = str(obj.get("action", "HOLD")).upper()
            conf = float(obj.get("confidence", 0.5))
            if action in ("BUY", "SELL", "HOLD"):
                return action, max(0.0, min(1.0, conf))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    up = response.upper()
    for a in ("BUY", "SELL", "HOLD"):
        if a in up:
            return a, 0.5
    return "HOLD", 0.5


class FinMemAgent(Strategy):
    name = "S2_finmem_agent"
    rebalance = "daily"

    def __init__(self, client, tickers: list[str]):
        self.client = client
        self.tickers = sorted(set(tickers))
        self.memory: dict[str, list[tuple[pd.Timestamp, str]]] = {tk: [] for tk in self.tickers}
        self.last_action: dict[str, str] = {}

    def state_dict(self) -> dict:
        return {"memory": self.memory, "last_action": self.last_action}

    def load_state_dict(self, state: dict) -> None:
        self.memory = state.get("memory", {tk: [] for tk in self.tickers})
        self.last_action = state.get("last_action", {})

    # ------------------------------------------------------------------ #

    def _retrieve(self, tk: str, t: pd.Timestamp) -> str:
        notes = []
        events = [e for e in self.memory.get(tk, []) if e[0] <= t]
        for horizon, keep in _LAYERS:
            lo = t - pd.Timedelta(days=horizon)
            layer = [txt for (d, txt) in events if d > lo][-keep:]
            notes.extend(layer)
        seen, uniq = set(), []
        for n in notes:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return "\n".join(f"- {n}" for n in uniq) if uniq else "- (no relevant memories)"

    def _reflect(self, tk: str, t, view) -> None:
        """Store yesterday's outcome vs. yesterday's action (uses only past data)."""
        act = self.last_action.get(tk)
        if act is None or tk not in view.prices.columns or len(view.prices) < 2:
            return
        p = view.prices[tk].dropna()
        if len(p) < 2:
            return
        r1 = (p.iloc[-1] / p.iloc[-2] - 1.0) * 100
        self.memory[tk].append(
            (t, f"[{t.date()}] reflection: yesterday I chose {act}; {tk} then moved {r1:+.2f}%")
        )

    # ------------------------------------------------------------------ #

    def decide(self, t, view) -> Decision:
        active = set(self.tickers)
        if view.members is not None:
            active &= set(view.members)
        news = view.news_window(tickers=active)
        raw: dict[str, float] = {}
        max_ts = None

        for tk in sorted(active):
            p = view.prices[tk].dropna() if tk in view.prices.columns else pd.Series(dtype=float)
            if len(p) < 21:
                continue
            r5 = (p.iloc[-1] / p.iloc[-6] - 1.0) * 100
            r20 = (p.iloc[-1] / p.iloc[-21] - 1.0) * 100

            self._reflect(tk, t, view)
            todays = news[news["ticker"] == tk] if len(news) else news
            headlines = "\n".join(f"- {h}" for h in todays["headline"]) if len(todays) else "- (no headlines today)"
            if len(todays):
                m = todays["timestamp_utc"].max()
                max_ts = m if max_ts is None or m > max_ts else max_ts
                digest = "; ".join(todays["headline"].head(3))
                self.memory[tk].append((t, f"[{t.date()}] news: {digest}"))

            prompt = ACTION_PROMPT.format(
                profile=PROFILE, date=t.date(), ticker=tk,
                r5=r5, r20=r20, memories=self._retrieve(tk, t), headlines=headlines,
            )
            action, conf = parse_action(self.client.complete(prompt))
            self.last_action[tk] = action
            if action == "BUY":
                raw[tk] = conf
            elif action == "SELL":
                raw[tk] = -conf

        gross = sum(abs(v) for v in raw.values())
        if gross > 1.0:
            raw = {k: v / gross for k, v in raw.items()}
        return Decision(raw, data_max_ts=max_ts)
