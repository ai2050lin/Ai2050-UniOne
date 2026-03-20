from __future__ import annotations

from stage56_learning_dynamics_terminal_form import build_learning_dynamics_terminal_form_summary


def test_learning_dynamics_terminal_form_positive() -> None:
    summary = build_learning_dynamics_terminal_form_summary()
    hm = summary["headline_metrics"]
    assert hm["terminal_seed"] > 0.0
    assert hm["terminal_feature"] > 0.0
    assert hm["terminal_structure"] > 0.0
    assert hm["terminal_global"] > 0.0
