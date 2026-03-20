try:
    from tests.codex.stage56_large_model_learning_equation_bridge import build_large_model_learning_equation_summary
except ModuleNotFoundError:
    from stage56_large_model_learning_equation_bridge import build_large_model_learning_equation_summary


def test_large_model_learning_equation_summary_nonnegative() -> None:
    summary = build_large_model_learning_equation_summary()
    assert summary["headline_metrics"]["large_formula_support"] >= 0.0
