from __future__ import annotations

from stage56_kernel_v4_validation import build_rows


def test_kernel_v4_validation_builds_general_targets() -> None:
    rows = build_rows(
        [
            {
                "style_structure_gain_term": -0.4,
                "core_balance_v3_term": 3.0,
                "logic_strictload_term": 0.5,
                "union_joint_adv": 0.2,
                "union_synergy_joint": 0.4,
                "strict_positive_synergy": 0.9,
            }
        ]
    )
    row = rows[0]
    assert abs(row["kernel_v4_term"] - 3.1) < 1e-9
    assert abs(row["general_mean_target"] - 0.3) < 1e-9
    assert abs(row["strictness_delta_vs_general"] - 0.6) < 1e-9
