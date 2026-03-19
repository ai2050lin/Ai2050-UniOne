from __future__ import annotations

from stage56_general_strict_dual_kernel import build_summary


def test_general_strict_dual_kernel_summary_collects_signs() -> None:
    summary = build_summary(
        {
            "sign_matrix": {
                "kernel_v4_term": {
                    "union_joint_adv": "positive",
                    "union_synergy_joint": "positive",
                    "strict_positive_synergy": "positive",
                }
            }
        },
        {
            "sign_matrix": {
                "strict_module_combined_term": {
                    "union_joint_adv": "positive",
                    "union_synergy_joint": "positive",
                    "strict_positive_synergy": "positive",
                },
                "strict_module_residual_term": {
                    "union_joint_adv": "negative",
                    "union_synergy_joint": "negative",
                    "strict_positive_synergy": "negative",
                },
            }
        },
    )
    assert summary["general_kernel_sign"]["union_joint_adv"] == "positive"
    assert summary["strict_module_combined_sign"]["strict_positive_synergy"] == "positive"
    assert summary["strict_module_residual_sign"]["union_synergy_joint"] == "negative"
