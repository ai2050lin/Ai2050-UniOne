try:
    from tests.codex.stage56_large_model_native_variable_refinement import build_native_variable_summary
except ModuleNotFoundError:
    from stage56_large_model_native_variable_refinement import build_native_variable_summary


def test_native_variable_summary_balance_exists() -> None:
    summary = build_native_variable_summary()
    assert "headline_metrics" in summary
    assert "native_balance" in summary["headline_metrics"]

