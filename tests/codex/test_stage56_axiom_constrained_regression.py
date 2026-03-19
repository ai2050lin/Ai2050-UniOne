from __future__ import annotations

from stage56_axiom_constrained_regression import build_axiom_rows, build_summary


def test_build_axiom_rows_projects_features() -> None:
    rows = [
        {
            "atlas_static_proxy": -1.0,
            "offset_static_proxy": 0.2,
            "frontier_dynamic_proxy": 0.3,
            "logic_prototype_proxy": 0.4,
            "logic_fragile_bridge_proxy": 0.5,
            "syntax_constraint_conflict_proxy": 0.6,
            "window_hidden_proxy": 10.0,
            "window_mlp_proxy": 12.0,
            "style_control_proxy": 0.7,
            "logic_control_proxy": -0.8,
            "syntax_control_proxy": 0.9,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        }
    ]
    axiom_rows = build_axiom_rows(rows)
    assert axiom_rows[0]["atlas_axiom_feature"] == 0.0
    assert axiom_rows[0]["subfield_axiom_feature"] == 1.0
    assert axiom_rows[0]["window_axiom_feature"] == 11.0


def test_build_summary_returns_three_targets() -> None:
    row = {
        "atlas_static_proxy": 0.1,
        "offset_static_proxy": 0.2,
        "frontier_dynamic_proxy": 0.3,
        "logic_prototype_proxy": 0.4,
        "logic_fragile_bridge_proxy": 0.05,
        "syntax_constraint_conflict_proxy": 0.06,
        "window_hidden_proxy": 10.0,
        "window_mlp_proxy": 11.0,
        "style_control_proxy": 0.7,
        "logic_control_proxy": -0.8,
        "syntax_control_proxy": 0.9,
        "union_joint_adv": 0.2,
        "union_synergy_joint": 0.1,
        "strict_positive_synergy": 1.0,
    }
    summary = build_summary([row, dict(row, union_joint_adv=0.3, union_synergy_joint=0.0, strict_positive_synergy=0.0)])
    assert summary["record_type"] == "stage56_axiom_constrained_regression_summary"
    assert len(summary["fits"]) == 3
