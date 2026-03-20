try:
    from tests.codex.stage56_continuous_learning_ode import build_continuous_learning_ode_summary
except ModuleNotFoundError:
    from stage56_continuous_learning_ode import build_continuous_learning_ode_summary


def test_continuous_learning_ode_signs() -> None:
    summary = build_continuous_learning_ode_summary()
    hm = summary["headline_metrics"]
    assert hm["d_frontier"] > 0.0
    assert hm["d_boundary"] > 0.0
    assert hm["d_circuit"] > 0.0
    assert hm["d_atlas"] < 0.0

