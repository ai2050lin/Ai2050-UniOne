from __future__ import annotations

from stage56_fullsample_regression_runner import build_design_rows, build_summary


def test_build_design_rows_aggregates_axis_and_component_rows() -> None:
    pair_density_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "axis": "style",
            "role_alignment_compaction": 0.9,
            "role_alignment_coverage": 0.8,
            "role_asymmetry_compaction_l1": 0.1,
            "role_asymmetry_coverage_l1": 0.2,
            "pair_compaction_middle_mean": 0.3,
            "pair_coverage_middle_mean": 0.4,
        },
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "axis": "logic",
            "role_alignment_compaction": 0.7,
            "role_alignment_coverage": 0.6,
            "role_asymmetry_compaction_l1": 0.2,
            "role_asymmetry_coverage_l1": 0.3,
            "pair_compaction_middle_mean": 0.5,
            "pair_coverage_middle_mean": 0.6,
        },
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "axis": "syntax",
            "role_alignment_compaction": 0.5,
            "role_alignment_coverage": 0.4,
            "role_asymmetry_compaction_l1": 0.3,
            "role_asymmetry_coverage_l1": 0.4,
            "pair_compaction_middle_mean": 0.7,
            "pair_coverage_middle_mean": 0.8,
        },
    ]
    complete_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "component_label": "logic_prototype",
            "weight": 0.1,
            "hidden_window_center": 5.0,
            "mlp_window_center": 6.0,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": True,
        },
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "component_label": "logic_fragile_bridge",
            "weight": 0.05,
            "hidden_window_center": 4.0,
            "mlp_window_center": 5.0,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": True,
        },
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "component_label": "syntax_constraint_conflict",
            "weight": 0.08,
            "hidden_window_center": 3.0,
            "mlp_window_center": 4.0,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": True,
        },
    ]

    rows = build_design_rows(pair_density_rows, complete_rows)
    assert len(rows) == 1
    row = rows[0]
    assert row["atlas_static_proxy"] > 0.0
    assert row["frontier_dynamic_proxy"] > 0.0
    assert row["logic_prototype_proxy"] == 0.1


def test_build_summary_returns_three_target_fits() -> None:
    rows = [
        {
            "atlas_static_proxy": 0.5,
            "offset_static_proxy": 0.3,
            "frontier_dynamic_proxy": 0.6,
            "logic_prototype_proxy": 0.1,
            "logic_fragile_bridge_proxy": 0.02,
            "syntax_constraint_conflict_proxy": 0.08,
            "window_hidden_proxy": 5.0,
            "window_mlp_proxy": 6.0,
            "style_control_proxy": 0.2,
            "logic_control_proxy": 0.3,
            "syntax_control_proxy": 0.4,
            "union_joint_adv": 0.3,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        },
        {
            "atlas_static_proxy": 0.4,
            "offset_static_proxy": 0.2,
            "frontier_dynamic_proxy": 0.5,
            "logic_prototype_proxy": 0.05,
            "logic_fragile_bridge_proxy": 0.03,
            "syntax_constraint_conflict_proxy": 0.04,
            "window_hidden_proxy": 4.0,
            "window_mlp_proxy": 5.0,
            "style_control_proxy": 0.1,
            "logic_control_proxy": 0.2,
            "syntax_control_proxy": 0.3,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.05,
            "strict_positive_synergy": 0.0,
        },
    ]
    summary = build_summary(rows)
    assert summary["record_type"] == "stage56_fullsample_regression_runner_summary"
    assert summary["row_count"] == 2
    assert len(summary["fits"]) == 3
