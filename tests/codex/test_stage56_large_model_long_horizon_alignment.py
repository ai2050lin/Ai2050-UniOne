try:
    from tests.codex.stage56_large_model_long_horizon_alignment import build_long_horizon_alignment_summary
except ModuleNotFoundError:
    from stage56_large_model_long_horizon_alignment import build_long_horizon_alignment_summary


def test_long_horizon_alignment_summary_has_cases() -> None:
    summary = build_long_horizon_alignment_summary()
    assert summary["case_count"] >= 4
    assert summary["headline_metrics"]["ordered_case_ratio"] >= 0.0

