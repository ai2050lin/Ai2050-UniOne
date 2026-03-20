from __future__ import annotations

from stage56_pulse_to_network_learning_dynamics import build_pulse_to_network_learning_dynamics_summary


def test_pulse_to_network_learning_dynamics_positive() -> None:
    summary = build_pulse_to_network_learning_dynamics_summary()
    hm = summary["headline_metrics"]
    assert hm["learning_seed"] > 0.0
    assert hm["learning_feature"] > 0.0
    assert hm["learning_structure"] > 0.0
    assert hm["learning_global"] > 0.0
