from __future__ import annotations

from tests.codex.stage56_total_theory_bridge_expansion import build_total_theory_bridge_expansion_summary


def test_total_theory_bridge_expansion_is_positive() -> None:
    hm = build_total_theory_bridge_expansion_summary()["headline_metrics"]
    assert hm["dnn_to_brain_alignment"] > 0.0
    assert hm["brain_to_math_alignment"] > 0.0
    assert hm["math_to_intelligence_alignment"] > 0.0
    assert hm["total_bridge_strength_expanded"] > 0.0
