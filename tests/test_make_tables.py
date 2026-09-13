import pandas as pd

from mirage.experiments.reporting import current_window, dsr_trial, trading_bounds


def test_trading_bounds_use_actual_calendar_dates():
    calendar = pd.DatetimeIndex(["2025-08-29", "2025-09-02", "2025-09-03"])
    assert trading_bounds(calendar, "2025-09-01", "2025-09-03") == (
        "2025-09-02", "2025-09-03",
    )


def test_current_window_accepts_effective_bounds_only():
    bounds = {("2025-09-02", "2026-06-30"), ("2023-07-03", "2024-06-28")}
    assert current_window({"start": "2025-09-02", "end": "2026-06-30"}, bounds)
    assert not current_window({"start": "2025-07-01", "end": "2026-06-30"}, bounds)


def test_dsr_trials_exclude_ablation_and_exploratory_runs():
    assert dsr_trial({"analysis_role": "primary", "universe": "pit", "window": "eval"})
    assert dsr_trial({"analysis_role": "preregistered_prompt_arm", "universe": "pit"})
    assert not dsr_trial({"analysis_role": "exploratory_robustness", "universe": "pit"})
    assert not dsr_trial({"analysis_role": "primary", "universe": "static"})
    assert not dsr_trial({"analysis_role": "primary", "window": "inwindow"})
