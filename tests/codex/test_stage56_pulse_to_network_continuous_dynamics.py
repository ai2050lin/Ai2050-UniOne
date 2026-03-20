from __future__ import annotations

from stage56_pulse_to_network_continuous_dynamics import build_pulse_to_network_continuous_dynamics_summary


def test_pulse_to_network_continuous_dynamics_positive() -> None:
    summary = build_pulse_to_network_continuous_dynamics_summary()
    hm = summary["headline_metrics"]
    assert hm["d_seed"] > 0.0
    assert hm["d_feature"] > 0.0
    assert hm["d_structure"] > 0.0
    assert hm["d_global"] > 0.0
