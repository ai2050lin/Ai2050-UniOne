from __future__ import annotations

from stage56_icspb_closed_equation_draft import build_summary


def test_build_summary_contains_closed_equations() -> None:
    summary = build_summary(
        {"short_form": {}},
        {"sign_matrix": {"G_corpus_proxy": {"union_joint_adv": "positive"}}},
        {"signs": {"union_joint_adv": "positive"}},
        {"final_choice": {"feature": "strict_module_base_term"}},
    )
    eqs = dict(summary["closed_equations"])
    assert "U_general" in str(eqs["general_equation"])
    assert "U_strict" in str(eqs["strict_equation"])
