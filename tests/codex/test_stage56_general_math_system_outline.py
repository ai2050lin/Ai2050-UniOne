from __future__ import annotations

from stage56_general_math_system_outline import build_outline


def test_build_outline_answers_two_questions() -> None:
    out = build_outline()
    assert out["record_type"] == "stage56_general_math_system_outline"
    assert "单一的低层神经机制" in out["main_answer_1"]
    assert "更一般的数学体系" in out["main_answer_2"]
    assert len(out["proto_axioms"]) == 5
