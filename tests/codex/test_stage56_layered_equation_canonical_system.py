from __future__ import annotations

from stage56_layered_equation_canonical_system import build_summary


def test_layered_equation_canonical_system_passes_formal_and_channel_data() -> None:
    summary = build_summary(
        {"formal_equations": {"general_layer": "U_general(pair) = kernel_v4(pair)"}, "layer_stability": {}},
        {"sign_matrix": {"gd_drive_channel_term": {"union_joint_adv": "positive"}}},
    )
    assert summary["formal_equations"]["general_layer"] == "U_general(pair) = kernel_v4(pair)"
    assert summary["canonical_channels"]["gd_drive_channel_term"]["union_joint_adv"] == "positive"
