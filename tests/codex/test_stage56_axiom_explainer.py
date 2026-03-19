from __future__ import annotations

from stage56_axiom_explainer import build_axiom_explanations


def test_build_axiom_explanations_contains_six_axioms() -> None:
    out = build_axiom_explanations()
    assert out["record_type"] == "stage56_axiom_explainer_summary"
    assert len(out["axioms"]) == 6
    assert out["axioms"][0]["name"] == "局部图册公理"
