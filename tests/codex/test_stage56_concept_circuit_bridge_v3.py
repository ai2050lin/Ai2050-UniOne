from __future__ import annotations

from stage56_concept_circuit_bridge_v3 import build_concept_circuit_bridge_v3_summary


def test_concept_circuit_bridge_v3_balances_terms() -> None:
    summary = build_concept_circuit_bridge_v3_summary()
    hm = summary["headline_metrics"]
    assert hm["concept_circuit_balance_v3"] > 0.0
    assert hm["embed_balanced"] > hm["inhibit_balanced"]
    assert hm["bind_balanced"] > 0.0
