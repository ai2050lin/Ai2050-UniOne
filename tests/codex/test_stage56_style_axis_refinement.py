from __future__ import annotations

from stage56_style_axis_refinement import build_rows, build_summary


def test_build_rows_creates_style_subchannels() -> None:
    pair_density_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": True,
            "axes": {
                "style": {
                    "pair_compaction_middle_mean": 0.3,
                    "pair_coverage_middle_mean": 0.7,
                    "pair_delta_l2": 1.2,
                    "pair_delta_mean_abs": 0.4,
                    "role_alignment_compaction": 0.9,
                    "role_alignment_coverage": 0.8,
                }
            },
        }
    ]
    complete_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "component_label": "logic_prototype",
        }
    ]
    rows = build_rows(pair_density_rows, complete_rows)
    assert len(rows) == 1
    assert rows[0]["style_midfield"] == 0.5
    assert rows[0]["style_gap"] == 0.4


def test_build_summary_returns_sign_matrix() -> None:
    rows = [
        {
            "style_compaction_mid": 0.3,
            "style_coverage_mid": 0.7,
            "style_delta_l2": 1.2,
            "style_delta_mean_abs": 0.4,
            "style_role_align_compaction": 0.9,
            "style_role_align_coverage": 0.8,
            "style_midfield": 0.5,
            "style_alignment": 0.85,
            "style_reorder_pressure": 0.8,
            "style_gap": 0.4,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        },
        {
            "style_compaction_mid": 0.4,
            "style_coverage_mid": 0.6,
            "style_delta_l2": 1.1,
            "style_delta_mean_abs": 0.5,
            "style_role_align_compaction": 0.8,
            "style_role_align_coverage": 0.7,
            "style_midfield": 0.5,
            "style_alignment": 0.75,
            "style_reorder_pressure": 0.8,
            "style_gap": 0.2,
            "union_joint_adv": 0.1,
            "union_synergy_joint": 0.0,
            "strict_positive_synergy": 0.0,
        },
    ]
    summary = build_summary(rows)
    assert summary["record_type"] == "stage56_style_axis_refinement_summary"
    assert "sign_matrix" in summary
