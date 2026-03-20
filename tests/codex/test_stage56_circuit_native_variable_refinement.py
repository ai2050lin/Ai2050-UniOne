from __future__ import annotations

from stage56_circuit_native_variable_refinement import build_circuit_native_variable_refinement_summary


def test_circuit_native_variable_refinement_positive() -> None:
    summary = build_circuit_native_variable_refinement_summary()
    hm = summary["headline_metrics"]
    assert hm["native_attractor"] > hm["native_gate"]
    assert hm["circuit_native_margin"] > 0.0
