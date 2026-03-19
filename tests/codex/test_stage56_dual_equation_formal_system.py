from __future__ import annotations

from stage56_dual_equation_formal_system import build_summary


def test_dual_equation_formal_system_builds_three_layer_equations() -> None:
    summary = build_summary(
        {
            "general_layer": {"stable_signs": {"union_joint_adv": "positive"}},
            "strict_layer": {"final_choice": {"feature": "strict_module_base_term"}},
            "discriminator_layer": {"stable_signs": {"strict_positive_synergy": "positive"}},
        },
        {
            "sign_matrix": {
                "gs_coupling_term": {"union_joint_adv": "positive"},
                "gd_coupling_term": {"union_joint_adv": "negative"},
            }
        },
    )
    assert summary["formal_equations"]["general_layer"] == "U_general(pair) = kernel_v4(pair)"
    assert summary["layer_stability"]["strict_choice"]["feature"] == "strict_module_base_term"
    assert summary["coupling_signs"]["gd_coupling_term"]["union_joint_adv"] == "negative"
