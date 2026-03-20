from __future__ import annotations

from stage56_circuit_direct_refinement_v2 import build_circuit_direct_refinement_v2_summary


def test_circuit_direct_refinement_v2_positive() -> None:
    summary = build_circuit_direct_refinement_v2_summary()
    hm = summary["headline_metrics"]
    assert hm["direct_attractor_v2"] > hm["direct_gate_v2"]
    assert hm["direct_margin_v2"] > 0.0
