from __future__ import annotations

from stage56_higher_order_math_axioms import build_axioms


def test_build_axioms_contains_six_core_axioms() -> None:
    out = build_axioms()
    assert out["record_type"] == "stage56_higher_order_math_axioms_summary"
    assert len(out["axioms"]) == 6
    assert "更高阶数学体系" in out["math_possibility_answer"]
