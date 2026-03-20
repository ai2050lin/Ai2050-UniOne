from __future__ import annotations

from stage56_circuit_dynamics_bridge_v2 import build_circuit_dynamics_bridge_v2_summary


def test_circuit_dynamics_bridge_v2_positive() -> None:
    summary = build_circuit_dynamics_bridge_v2_summary()
    hm = summary["headline_metrics"]
    assert hm["attractor_loading"] > 0.0
    assert hm["circuit_dynamic_margin"] > 0.0
