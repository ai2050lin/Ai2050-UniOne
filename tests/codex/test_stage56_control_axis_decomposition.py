from __future__ import annotations

from stage56_control_axis_decomposition import build_rows, build_summary


def test_build_rows_extracts_nested_axis_channels() -> None:
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
                "style": {"pair_compaction_middle_mean": 0.3, "pair_coverage_middle_mean": 0.4},
                "logic": {"pair_compaction_middle_mean": 0.5, "pair_coverage_middle_mean": 0.6},
                "syntax": {"pair_compaction_middle_mean": 0.7, "pair_coverage_middle_mean": 0.8},
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
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": True,
        }
    ]
    rows = build_rows(pair_density_rows, complete_rows)
    assert len(rows) == 1
    assert rows[0]["logic_compaction_mid"] == 0.5
    assert rows[0]["syntax_coverage_mid"] == 0.8


def test_build_summary_produces_three_fits() -> None:
    rows = [
        {
            "style_compaction_mid": 0.2,
            "style_coverage_mid": 0.3,
            "style_delta_l2": 1.0,
            "style_delta_mean_abs": 0.1,
            "style_role_align_compaction": 0.9,
            "style_role_align_coverage": 0.9,
            "logic_compaction_mid": 0.4,
            "logic_coverage_mid": 0.5,
            "logic_delta_l2": 1.1,
            "logic_delta_mean_abs": 0.2,
            "logic_role_align_compaction": 0.8,
            "logic_role_align_coverage": 0.8,
            "syntax_compaction_mid": 0.6,
            "syntax_coverage_mid": 0.7,
            "syntax_delta_l2": 1.2,
            "syntax_delta_mean_abs": 0.3,
            "syntax_role_align_compaction": 0.7,
            "syntax_role_align_coverage": 0.7,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        },
        {
            "style_compaction_mid": 0.3,
            "style_coverage_mid": 0.4,
            "style_delta_l2": 1.2,
            "style_delta_mean_abs": 0.2,
            "style_role_align_compaction": 0.8,
            "style_role_align_coverage": 0.8,
            "logic_compaction_mid": 0.5,
            "logic_coverage_mid": 0.6,
            "logic_delta_l2": 1.3,
            "logic_delta_mean_abs": 0.3,
            "logic_role_align_compaction": 0.7,
            "logic_role_align_coverage": 0.7,
            "syntax_compaction_mid": 0.7,
            "syntax_coverage_mid": 0.8,
            "syntax_delta_l2": 1.4,
            "syntax_delta_mean_abs": 0.4,
            "syntax_role_align_compaction": 0.6,
            "syntax_role_align_coverage": 0.6,
            "union_joint_adv": 0.3,
            "union_synergy_joint": 0.2,
            "strict_positive_synergy": 0.0,
        },
    ]
    summary = build_summary(rows)
    assert summary["record_type"] == "stage56_control_axis_decomposition_summary"
    assert len(summary["fits"]) == 3
