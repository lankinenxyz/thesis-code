"""S3: one-request-per-ticker-day baseline with no memory or persona.

S3 receives the same per-ticker price and news inputs as S2. S3-P1 changes only
the decision sentence, and S3-A masks explicit identifiers of the focal issuer.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .base import Decision, Strategy
from .finmem_agent import parse_action

MAX_WORKERS = 3

NAIVE_PROMPT = """Today is {date}. Consider the stock {ticker}.

Recent price action:
- 5-day return: {r5:.2f}%
- 20-day return: {r20:.2f}%

Today's headlines for {ticker}:
{headlines}

Should a trader BUY, SELL, or HOLD this stock for the next trading day?
Respond with a single JSON object on one line:
{{"action": "BUY" | "SELL" | "HOLD", "confidence": <number 0..1>, "reason": "<one short sentence>"}}"""

NAIVE_PROMPT_P1 = """Today is {date}. Consider the stock {ticker}.

Recent price action:
- 5-day return: {r5:.2f}%
- 20-day return: {r20:.2f}%

Today's headlines for {ticker}:
{headlines}

Based only on these price statistics and headlines, select BUY, SELL, or HOLD for this stock over the next trading day.
Respond with a single JSON object on one line:
{{"action": "BUY" | "SELL" | "HOLD", "confidence": <number 0..1>, "reason": "<one short sentence>"}}"""

PROMPTS = {"original": NAIVE_PROMPT, "p1": NAIVE_PROMPT_P1}

# Frozen focal-issuer aliases for the 2025-09-01 evaluation sample. The mask
# removes explicit identifiers, not indirect clues such as products or events.
ISSUER_ALIASES = {
    "ARE": ("Alexandria Real Estate Equities",),
    "ATO": ("Atmos Energy",),
    "AVY": ("Avery Dennison",),
    "BLDR": ("Builders FirstSource", "Builders First Source"),
    "CI": ("The Cigna Group", "Cigna", "Evernorth"),
    "CMG": ("Chipotle Mexican Grill", "Chipotle"),
    "FDS": ("FactSet Research Systems", "FactSet"),
    "GEN": ("Gen Digital", "NortonLifeLock", "Norton", "Avast"),
    "GM": ("General Motors", "Chevrolet", "Cadillac", "Buick"),
    "GOOG": ("Alphabet", "Google"),
    "HII": ("Huntington Ingalls Industries", "Huntington Ingalls", "Ingalls Shipbuilding"),
    "HLT": ("Hilton Worldwide", "Hilton"),
    "IRM": ("Iron Mountain",),
    "IVZ": ("Invesco",),
    "JBL": ("Jabil",),
    "LDOS": ("Leidos",),
    "MO": ("Altria Group", "Altria", "Philip Morris USA"),
    "MSI": ("Motorola Solutions",),
    "NOW": ("ServiceNow",),
    "OMC": ("Omnicom Group", "Omnicom"),
    "PAYX": ("Paychex",),
    "PH": ("Parker-Hannifin", "Parker Hannifin"),
    "PLTR": ("Palantir Technologies", "Palantir"),
    "PTC": ("PTC Inc", "PTC Incorporated"),
    "REG": ("Regency Centers",),
    "SO": ("The Southern Company", "Southern Company"),
    "SOLV": ("Solventum",),
    "SPG": ("Simon Property Group",),
    "UNH": ("UnitedHealth Group", "UnitedHealth", "UnitedHealthcare", "Optum"),
    "VRTX": ("Vertex Pharmaceuticals",),
}
ANONYMIZATION_TOKEN = "[COMPANY]"
ANONYMIZATION_VERSION = "focal_explicit_identifiers_v1"


def anonymize_headline(headline: str, ticker: str) -> str:
    """Mask the focal ticker and frozen issuer aliases in one headline."""
    text = re.sub(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", ANONYMIZATION_TOKEN, headline)
    aliases = sorted(ISSUER_ALIASES.get(ticker, ()), key=len, reverse=True)
    for alias in aliases:
        text = re.sub(
            rf"(?<!\w){re.escape(alias)}(?!\w)",
            ANONYMIZATION_TOKEN,
            text,
            flags=re.IGNORECASE,
        )
    return text


class NaivePrompt(Strategy):
    rebalance = "daily"

    def __init__(self, client, tickers: list[str], prompt_variant: str = "original",
                 anonymized: bool = False):
        if prompt_variant not in PROMPTS:
            raise ValueError(f"unknown prompt variant {prompt_variant}")
        if anonymized and prompt_variant != "original":
            raise ValueError("anonymization and prompt paraphrasing are separate arms")
        self.client = client
        self.tickers = sorted(set(tickers))
        self.prompt_variant = prompt_variant
        self.prompt_template = PROMPTS[prompt_variant]
        self.anonymized = anonymized
        if anonymized:
            self.name = "S3_naive_prompt_anonymized"
            self.analysis_role = "exploratory_robustness"
        elif prompt_variant == "p1":
            self.name = "S3_naive_prompt_paraphrase_p1"
            self.analysis_role = "preregistered_prompt_arm"
        else:
            self.name = "S3_naive_prompt"
            self.analysis_role = "primary"

    def _case(self, t, tk: str, view, news: pd.DataFrame) -> dict | None:
        p = view.prices[tk].dropna() if tk in view.prices.columns else pd.Series(dtype=float)
        if len(p) < 21:
            return None
        todays = news[news["ticker"] == tk] if len(news) else news
        headline_values = list(todays["headline"]) if len(todays) else []
        if self.anonymized:
            headline_values = [anonymize_headline(h, tk) for h in headline_values]
        headlines = "\n".join(f"- {h}" for h in headline_values) or "- (no headlines today)"
        return {
            "ticker": tk,
            "display_ticker": ANONYMIZATION_TOKEN if self.anonymized else tk,
            "r5": (p.iloc[-1] / p.iloc[-6] - 1.0) * 100,
            "r20": (p.iloc[-1] / p.iloc[-21] - 1.0) * 100,
            "headlines": headlines,
            "max_ts": todays["timestamp_utc"].max() if len(todays) else None,
        }

    def _decide_one(self, t, case: dict) -> tuple[str, tuple[str, float]]:
        prompt = self.prompt_template.format(
            date=t.date(), ticker=case["display_ticker"],
            r5=case["r5"], r20=case["r20"], headlines=case["headlines"],
        )
        return case["ticker"], parse_action(self.client.complete(prompt))

    def decide(self, t, view) -> Decision:
        active = set(self.tickers)
        if view.members is not None:
            active &= set(view.members)
        news = view.news_window(tickers=active)
        raw: dict[str, float] = {}
        max_ts = None
        cases = []

        for tk in sorted(active):
            case = self._case(t, tk, view, news)
            if case is None:
                continue
            if case["max_ts"] is not None:
                m = case["max_ts"]
                max_ts = m if max_ts is None or m > max_ts else max_ts
            cases.append(case)

        actions: dict[str, tuple[str, float]] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(self._decide_one, t, case) for case in cases]
            for fut in as_completed(futures):
                tk, action = fut.result()
                actions[tk] = action

        for tk in sorted(actions):
            action, conf = actions[tk]
            if action == "BUY":
                raw[tk] = conf
            elif action == "SELL":
                raw[tk] = -conf

        gross = sum(abs(v) for v in raw.values())
        if gross > 1.0:
            raw = {k: v / gross for k, v in raw.items()}
        return Decision(raw, data_max_ts=max_ts)
