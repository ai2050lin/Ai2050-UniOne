from __future__ import annotations

from stage56_generalized_formula_refinement import build_summary


def test_build_summary_promotes_base_and_selective_operators() -> None:
    generalized_summary = {"generalized_formulas": {"layer_state_vector": "z(pair) = [G, S, D]^T"}}
    load_summary = {
        "sign_matrix": {
            "load_mean_term": {
                "union_joint_adv": "negative",
                "union_synergy_joint": "negative",
                "strict_positive_synergy": "negative",
            },
            "load_contrast_term": {
                "union_joint_adv": "negative",
                "union_synergy_joint": "negative",
                "strict_positive_synergy": "positive",
            },
        }
    }
    summary = build_summary(generalized_summary, load_summary)
    refined = dict(summary["refined_formulas"])
    assert "L_base" in str(refined["general_state_observation"])
    assert "L_select" in str(refined["strict_state_observation"])
