from __future__ import annotations

from stage56_circuit_native_direct_measure import build_circuit_native_direct_measure_summary


def test_circuit_native_direct_measure_positive() -> None:
    summary = build_circuit_native_direct_measure_summary()
    hm = summary["headline_metrics"]
    assert hm["direct_attractor_measure"] > hm["direct_gate_measure"]
    assert hm["direct_circuit_margin"] > 0.0
