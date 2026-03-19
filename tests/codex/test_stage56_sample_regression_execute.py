from __future__ import annotations

from stage56_sample_regression_execute import build_execute_summary


def test_build_execute_summary_wraps_all_three_requested_blocks() -> None:
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
    out = build_execute_summary(pair_density_rows, complete_rows)
    assert out["record_type"] == "stage56_sample_regression_execute_summary"
    assert out["row_count"] == 1
    assert "control_summary" in out
    assert "family_summary" in out
