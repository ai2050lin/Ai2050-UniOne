from __future__ import annotations

from stage56_layered_equation_short_form import build_summary


def test_build_summary_exposes_short_form() -> None:
    generalized_refinement = {"refined_formulas": {"general_state_observation": "x"}}
    strict_select_summary = {
        "sign_matrix": {
            "load_contrast_term": {
                "strict_positive_synergy": "positive",
                "strictness_delta_vs_union": "positive",
            }
        }
    }
    formal_summary = {
        "layer_stability": {
            "strict_choice": {"feature": "strict_module_base_term"},
        }
    }
    summary = build_summary(generalized_refinement, strict_select_summary, formal_summary)
    short_form = dict(summary["short_form"])
    assert "U_general" in str(short_form["general_short_form"])
    assert "L_select" in str(short_form["strict_short_form"])
