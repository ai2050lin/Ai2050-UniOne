from __future__ import annotations

from stage56_dual_equation_layered import build_summary


def test_dual_equation_layered_builds_three_layers() -> None:
    summary = build_summary(
        {"signs": {"general_mean_target": "positive"}},
        {"final_choice": {"feature": "strict_module_base_term"}},
        {"signs": {"strictness_delta_vs_union": "positive"}},
    )
    assert summary["general_layer"]["equation"] == "U_general = kernel_v4"
    assert summary["strict_layer"]["final_choice"]["feature"] == "strict_module_base_term"
    assert summary["discriminator_layer"]["stable_signs"]["strictness_delta_vs_union"] == "positive"
