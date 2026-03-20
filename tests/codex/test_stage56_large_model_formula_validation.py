try:
    from tests.codex.stage56_large_model_formula_validation import build_formula_validation_summary
except ModuleNotFoundError:
    from stage56_large_model_formula_validation import build_formula_validation_summary


def test_formula_validation_summary_score_nonnegative() -> None:
    summary = build_formula_validation_summary()
    assert summary["headline_metrics"]["formula_support_score"] >= 0.0
