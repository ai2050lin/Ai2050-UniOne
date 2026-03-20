from __future__ import annotations

from stage56_concept_circuit_bridge_v2 import build_concept_circuit_bridge_v2_summary


def test_concept_circuit_bridge_v2_is_positive() -> None:
    summary = build_concept_circuit_bridge_v2_summary()
    hm = summary["headline_metrics"]

    assert hm["seed_circuit_term"] > 0.0
    assert hm["embed_circuit_term"] > 0.0
    assert hm["concept_circuit_margin_v2"] > hm["inhibit_circuit_term"]
