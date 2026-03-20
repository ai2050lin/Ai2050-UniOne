try:
    from tests.codex.stage56_cross_scale_learning_equation_unification import build_cross_scale_unification_summary
except ModuleNotFoundError:
    from stage56_cross_scale_learning_equation_unification import build_cross_scale_unification_summary


def test_cross_scale_summary_has_gap() -> None:
    summary = build_cross_scale_unification_summary()
    assert summary["headline_metrics"]["mean_absolute_gap"] >= 0.0

