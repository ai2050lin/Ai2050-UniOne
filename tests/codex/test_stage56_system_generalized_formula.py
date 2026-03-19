from __future__ import annotations

from stage56_system_generalized_formula import build_summary


def test_build_summary_extracts_generalized_invariants() -> None:
    kernel_summary = {
        "signs": {
            "union_joint_adv": "positive",
            "union_synergy_joint": "positive",
        }
    }
    gap_summary = {}
    formal_summary = {
        "layer_stability": {
            "discriminator": {
                "strictness_delta_vs_union": "positive",
                "strictness_delta_vs_synergy": "positive",
            },
            "strict_choice": {
                "feature": "strict_module_base_term",
            },
        }
    }
    canonical_summary = {
        "canonical_channels": {
            "gd_drive_channel_term": {
                "union_joint_adv": "positive",
                "union_synergy_joint": "positive",
            },
            "gs_load_channel_term": {
                "union_joint_adv": "positive",
                "union_synergy_joint": "negative",
            },
            "sd_load_channel_term": {
                "union_joint_adv": "negative",
                "union_synergy_joint": "positive",
            },
        }
    }

    summary = build_summary(kernel_summary, gap_summary, formal_summary, canonical_summary)

    invariants = dict(summary["invariants"])
    assert invariants["general_kernel_positive"] is True
    assert invariants["discriminator_positive"] is True
    assert invariants["gd_drive_positive"] is True
    assert invariants["gs_load_target_specific"] is True
    assert invariants["sd_load_target_specific"] is True
    assert invariants["strict_choice_target_specific"] is True

    formulas = dict(summary["generalized_formulas"])
    assert "y_t(pair)" in str(formulas["generalized_observation"])
    assert "L_t(gs, sd)" in str(formulas["load_operator"])
