from __future__ import annotations

from stage56_dual_gap_classifier import build_rows


def test_dual_gap_classifier_builds_strictness_deltas() -> None:
    rows = build_rows(
        [
            {
                "style_structure_gain_term": -0.4,
                "core_balance_v3_term": 3.0,
                "logic_strictload_term": 0.5,
                "strict_load_term": 1.2,
                "union_joint_adv": 0.2,
                "union_synergy_joint": 0.4,
                "strict_positive_synergy": 0.9,
            }
        ]
    )
    row = rows[0]
    assert abs(row["dual_gap_final_term"] - 1.9) < 1e-9
    assert abs(row["strictness_delta_vs_union"] - 0.7) < 1e-9
    assert abs(row["strictness_delta_vs_synergy"] - 0.5) < 1e-9
