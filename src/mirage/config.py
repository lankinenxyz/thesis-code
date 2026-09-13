"""Central configuration. Every experimental constant lives here, per the pre-registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
LLM_CACHE_PATH = DATA_DIR / "llm_cache.sqlite"

# --- Evaluation windows (original design plus amendments A1-A2) --------------
EVAL_START = "2025-09-01"   # strictly after pinned model knowledge cutoffs
EVAL_END = "2026-06-30"
# Ablation in-knowledge-window period (inside pretraining data of pinned models)
INWINDOW_START = "2023-07-01"
INWINDOW_END = "2024-06-30"
# Price history start (momentum needs 12m lookback + buffer before INWINDOW_START)
HISTORY_START = "2022-01-01"

# --- Universe -----------------------------------------------------------------
UNIVERSE_INDEX = "sp500"
LLM_SAMPLE_SIZE = 30       # LLM strategies run on a fixed random sample (cost control)
LLM_SAMPLE_SEED = 42
LLM_SAMPLE_DATE = EVAL_START
INWINDOW_SAMPLE_DATE = INWINDOW_START

# --- Costs (pre-registered grid, bps per side, applied to turnover) -----------
COST_GRID_BPS = (0, 10, 25, 50)
HEADLINE_COST_BPS = 25

# --- Primary model fixed by amendment A2; record it in every results file -----
@dataclass(frozen=True)
class ModelSpec:
    provider: str          # "anthropic" | "openai" | "opencode" | "mock"
    model: str             # exact snapshot id
    knowledge_cutoff: str  # vendor-documented cutoff (YYYY-MM), verify at run time

PRIMARY_MODEL = ModelSpec("opencode", "azure/gpt-5.4-mini", "2025-08")
ANTHROPIC_MODEL = ModelSpec("anthropic", "claude-sonnet-5", "2025-01")
SECONDARY_MODEL = ModelSpec("openai", "gpt-4o-2024-11-20", "2023-10")
OPENCODE_MODEL = PRIMARY_MODEL
MOCK_MODEL = ModelSpec("mock", "mock-deterministic-v1", "n/a")

# --- Statistics ----------------------------------------------------------------
TRADING_DAYS = 252
ROLLING_WINDOW = 63        # trading days (~3 months)
ROLLING_STEP = 21
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_BLOCK = 10       # mean block length, days
DSR_NUM_TRIALS = 8         # pre-registered variant count
RISK_FREE_ANNUAL = 0.0     # excess returns vs 0; documented simplification

# --- Momentum (B2) --------------------------------------------------------------
MOM_LOOKBACK = 252
MOM_SKIP = 21
MOM_TOP_FRACTION = 0.10
MOM_MIN_OBS = 200

# --- Strategy run defaults ------------------------------------------------------
@dataclass
class RunSpec:
    strategy: str
    start: str = EVAL_START
    end: str = EVAL_END
    cost_bps: tuple = COST_GRID_BPS
    model: ModelSpec = field(default_factory=lambda: MOCK_MODEL)
    universe: str = "pit"   # "pit" (point-in-time) | "static" (ablation arm)
    label: str = ""
