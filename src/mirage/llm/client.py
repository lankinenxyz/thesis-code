"""LLM client: anthropic / openai / opencode / mock providers behind one interface.

- temperature 0 everywhere (pre-registered).
- Every call goes through the sqlite cache; a completed run never re-bills.
- The mock provider is DETERMINISTIC and exists only to validate the pipeline
  end-to-end without API costs. Mock outputs must never be reported as findings.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time

from ..config import LLM_CACHE_PATH, ModelSpec
from .cache import PromptCache

TEMPERATURE = 0.0
MAX_TOKENS = 200


class LLMClient:
    def __init__(self, spec: ModelSpec, cache_path=None):
        self.spec = spec
        self.cache = PromptCache(cache_path or LLM_CACHE_PATH)
        self.calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()
        self._backend = None

    def complete(self, prompt: str) -> str:
        key = PromptCache.make_key(self.spec.provider, self.spec.model, TEMPERATURE, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
            return cached
        response = self._call(prompt)
        self.cache.put(key, self.spec.provider, self.spec.model, prompt, response)
        with self._lock:
            self.calls += 1
        return response

    # ------------------------------------------------------------------ #

    def _call(self, prompt: str) -> str:
        if self.spec.provider == "mock":
            return _mock_response(prompt)
        if self.spec.provider == "anthropic":
            return self._call_anthropic(prompt)
        if self.spec.provider == "openai":
            return self._call_openai(prompt)
        if self.spec.provider == "opencode":
            return self._call_opencode(prompt)
        raise ValueError(f"Unknown provider {self.spec.provider}")

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic

        if self._backend is None:
            self._backend = anthropic.Anthropic()
        for attempt in range(5):
            try:
                msg = self._backend.messages.create(
                    model=self.spec.model, max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            except anthropic.RateLimitError:
                time.sleep(2 ** attempt)
        raise RuntimeError("anthropic: rate-limited after retries")

    def _call_openai(self, prompt: str) -> str:
        import openai

        if self._backend is None:
            self._backend = openai.OpenAI()
        for attempt in range(5):
            try:
                out = self._backend.chat.completions.create(
                    model=self.spec.model, max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                return out.choices[0].message.content or ""
            except openai.RateLimitError:
                time.sleep(2 ** attempt)
        raise RuntimeError("openai: rate-limited after retries")

    def _call_opencode(self, prompt: str) -> str:
        wrapped = (
            "You are being called as a deterministic API backend for a research "
            "backtest. Do not use tools, inspect files, or explain the task. "
            "Return only the requested completion text.\n\n"
            f"{prompt}"
        )
        errors = []
        for attempt in range(5):
            try:
                cmd = ["opencode", "run", "--pure", "--model", self.spec.model]
                variant = os.environ.get("MIRAGE_OPENCODE_VARIANT")
                if variant:
                    cmd.extend(["--variant", variant])
                cmd.append(wrapped)
                out = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                return _clean_opencode_output(out.stdout)
            except subprocess.CalledProcessError as e:
                errors.append((e.stderr or e.stdout or str(e)).strip())
                time.sleep(2 ** attempt)
            except subprocess.TimeoutExpired as e:
                errors.append(str(e))
                time.sleep(2 ** attempt)
        detail = errors[-1] if errors else "unknown error"
        raise RuntimeError(f"opencode: failed after retries: {detail}")


# --------------------------------------------------------------------------- #
# Deterministic mock (pipeline validation only)
# --------------------------------------------------------------------------- #

_POS_WORDS = ("beats", "record", "upgrade", "raises", "wins", "strong", "buyback", "higher")
_NEG_WORDS = ("misses", "probe", "downgrade", "cuts", "recall", "weakening", "lawsuit", "rise")  # "costs expected to rise"


def _mock_response(prompt: str) -> str:
    p = prompt.lower()
    if "keyed by ticker" in p:
        out = {}
        parts = re.split(r"\n(?=Ticker:\s*)", prompt)
        for part in parts:
            mt = re.search(r"Ticker:\s*([A-Z0-9.-]+)", part)
            mr = re.search(r"20-day return:\s*(-?\d+(?:\.\d+)?)%", part)
            if not mt or not mr:
                continue
            r20 = float(mr.group(1))
            if r20 > 2.0:
                out[mt.group(1)] = {"action": "BUY", "confidence": 0.6, "reason": "positive momentum"}
            elif r20 < -2.0:
                out[mt.group(1)] = {"action": "SELL", "confidence": 0.6, "reason": "negative momentum"}
            else:
                out[mt.group(1)] = {"action": "HOLD", "confidence": 0.5, "reason": "no clear signal"}
        return json.dumps(out, separators=(",", ":"))
    if "headline:" in p:  # S1-style sentiment prompt
        headline = p.split("headline:", 1)[1]
        pos = any(w in headline for w in _POS_WORDS)
        neg = any(w in headline for w in _NEG_WORDS)
        if pos and not neg:
            return "YES\nThe headline reports clearly positive developments."
        if neg and not pos:
            return "NO\nThe headline reports clearly negative developments."
        return "UNKNOWN\nThe headline is not clearly directional."
    # S2/S3-style action prompt: deterministic momentum-following on the 20d figure
    m = re.search(r"20-day return:\s*(-?\d+(?:\.\d+)?)%", prompt)
    if m:
        r20 = float(m.group(1))
        if r20 > 2.0:
            return '{"action": "BUY", "confidence": 0.6, "reason": "positive momentum"}'
        if r20 < -2.0:
            return '{"action": "SELL", "confidence": 0.6, "reason": "negative momentum"}'
    return '{"action": "HOLD", "confidence": 0.5, "reason": "no clear signal"}'


def _clean_opencode_output(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].lstrip().startswith("> "):
        lines = lines[1:]
    return "\n".join(lines).strip()
