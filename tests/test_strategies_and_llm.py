import numpy as np
import pandas as pd
import pytest

from mirage.backtest.engine import run_backtest
from mirage.backtest.view import MarketData, MarketView
from mirage.config import MOCK_MODEL
from mirage.data.leakage import audit_decision_log
from mirage.data.news import synthetic_news
from mirage.llm.client import LLMClient, _clean_opencode_output, _mock_response
from mirage.strategies.finmem_agent import parse_action
from mirage.strategies.momentum import Momentum
from mirage.strategies.naive_prompt import (
    ANONYMIZATION_TOKEN,
    NAIVE_PROMPT,
    NAIVE_PROMPT_P1,
    NaivePrompt,
    anonymize_headline,
)
from mirage.strategies.news_sentiment import NewsSentiment, parse_llt


@pytest.fixture
def tmp_client(tmp_path):
    return LLMClient(MOCK_MODEL, cache_path=tmp_path / "cache.sqlite")


def test_parse_llt():
    assert parse_llt("YES\nGood news.") == 1
    assert parse_llt('  "NO", clearly bad') == -1
    assert parse_llt("UNKNOWN\n...") == 0
    assert parse_llt("gibberish") == 0


def test_parse_action():
    assert parse_action('{"action": "BUY", "confidence": 0.8, "reason": "x"}') == ("BUY", 0.8)
    assert parse_action("I would SELL here") == ("SELL", 0.5)
    assert parse_action("no idea")[0] == "HOLD"
    assert parse_action('{"action": "BUY", "confidence": 7}') == ("BUY", 1.0)  # clamped


def test_mock_is_deterministic_and_cached(tmp_path):
    c = LLMClient(MOCK_MODEL, cache_path=tmp_path / "c.sqlite")
    p = "Headline: ACME beats quarterly earnings estimates"
    r1, r2 = c.complete(p), c.complete(p)
    assert r1 == r2
    assert c.calls == 1 and c.cache_hits == 1
    assert _mock_response(p).startswith("YES")


def test_clean_opencode_output_strips_header():
    raw = "\x1b[0m\n> build · gpt-4o\n\x1b[0m\nYES\nGood news.\n"
    assert _clean_opencode_output(raw) == "YES\nGood news."


def test_synthetic_news_deterministic():
    days = pd.bdate_range("2025-07-01", periods=5)
    a = synthetic_news(["AAA", "BBB"], days)
    b = synthetic_news(["AAA", "BBB"], days)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["synthetic_label"].unique()) <= {-1, 0, 1}


def test_s1_end_to_end_no_leakage(tmp_client):
    days = pd.bdate_range("2025-06-02", periods=30)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (30, 3)), axis=0),
        index=days, columns=["AAA", "BBB", "CCC"],
    )
    news = synthetic_news(["AAA", "BBB", "CCC"], days)
    market = MarketData(prices, news=news)
    strat = NewsSentiment(tmp_client, ["AAA", "BBB", "CCC"])
    res = run_backtest(strat, market, days[5], days[-1], cost_grid=(0, 25))
    assert len(res.decisions) > 0
    assert len(audit_decision_log(res.decisions)) == 0  # leakage audit clean
    # long-short legs each normalized to <= 1 in absolute sum
    if len(res.weights):
        gross = res.weights.abs().sum(axis=1)
        assert (gross <= 2.0 + 1e-9).all()


def test_s3_calls_once_per_ticker_day(tmp_client):
    days = pd.bdate_range("2025-06-02", periods=24)
    up = 100 * (1.003 ** np.arange(24))
    down = 100 * (0.997 ** np.arange(24))
    flat = np.full(24, 100.0)
    prices = pd.DataFrame({"AAA": up, "BBB": down, "CCC": flat}, index=days)
    market = MarketData(prices, news=synthetic_news(["AAA", "BBB", "CCC"], days))
    strat = NaivePrompt(tmp_client, ["AAA", "BBB", "CCC"])
    res = run_backtest(strat, market, days[20], days[-1], cost_grid=(0, 25))
    assert len(res.decisions) == 3
    assert tmp_client.calls == 9


def test_s3_prompt_variants_are_separate_arms(tmp_client):
    canonical = NaivePrompt(tmp_client, ["GOOG"])
    paraphrase = NaivePrompt(tmp_client, ["GOOG"], prompt_variant="p1")
    assert canonical.prompt_template == NAIVE_PROMPT
    assert paraphrase.prompt_template == NAIVE_PROMPT_P1
    assert canonical.name != paraphrase.name
    with pytest.raises(ValueError):
        NaivePrompt(tmp_client, ["GOOG"], prompt_variant="p1", anonymized=True)


def test_anonymize_headline_masks_only_focal_identifiers():
    text = anonymize_headline("Google and Microsoft discuss GOOG services", "GOOG")
    assert text == f"{ANONYMIZATION_TOKEN} and Microsoft discuss {ANONYMIZATION_TOKEN} services"
    assert anonymize_headline("Shares are now higher", "NOW") == "Shares are now higher"


def test_s3_anonymized_prompt_keeps_internal_ticker():
    class RecordingClient:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt):
            self.prompts.append(prompt)
            return '{"action":"BUY","confidence":0.8,"reason":"x"}'

    days = pd.bdate_range("2025-06-02", periods=24)
    prices = pd.DataFrame({"GOOG": 100 * (1.003 ** np.arange(24))}, index=days)
    news = pd.DataFrame({
        "timestamp_utc": [pd.Timestamp("2025-07-03 15:00", tz="UTC")],
        "ticker": ["GOOG"],
        "headline": ["Google expands GOOG cloud services"],
    })
    client = RecordingClient()
    market = MarketData(prices, news=news)
    view = MarketView(market, days[-1], days[-2])
    decision = NaivePrompt(client, ["GOOG"], anonymized=True).decide(days[-1], view)
    assert set(decision.weights) == {"GOOG"}
    assert "GOOG" not in client.prompts[0]
    assert "Google" not in client.prompts[0]
    assert ANONYMIZATION_TOKEN in client.prompts[0]


def test_momentum_picks_trending():
    days = pd.bdate_range("2023-01-02", periods=300)
    up = 100 * (1.002 ** np.arange(300))
    flat = np.full(300, 100.0)
    down = 100 * (0.999 ** np.arange(300))
    cols = {f"UP{i}": up * (1 + i * 0.001) for i in range(10)}
    cols |= {f"FL{i}": flat for i in range(10)}
    cols |= {f"DN{i}": down for i in range(10)}
    prices = pd.DataFrame(cols, index=days)
    market = MarketData(prices)
    view = MarketView(market, days[-1], days[-2])
    d = Momentum().decide(days[-1], view)
    assert d is not None
    assert all(tk.startswith("UP") for tk in d.weights)
    assert sum(d.weights.values()) == pytest.approx(1.0)


def test_leakage_audit_catches_future_data():
    log = pd.DataFrame({
        "decision_day": [pd.Timestamp("2025-07-01")],
        "data_max_ts": [pd.Timestamp("2025-07-02 14:00", tz="UTC")],  # next day!
    })
    assert len(audit_decision_log(log)) == 1
