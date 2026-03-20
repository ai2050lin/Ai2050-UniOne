try:
    from tests.codex.stage56_large_model_long_horizon_stability import build_long_horizon_stability_summary
except ModuleNotFoundError:
    from stage56_large_model_long_horizon_stability import build_long_horizon_stability_summary


def test_long_horizon_stability_summary_has_cases() -> None:
    summary = build_long_horizon_stability_summary()
    assert summary["case_count"] >= 6
    assert summary["headline_metrics"]["best_balance_case"]

