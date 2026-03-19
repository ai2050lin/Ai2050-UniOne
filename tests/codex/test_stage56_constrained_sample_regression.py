from __future__ import annotations

from stage56_constrained_sample_regression import apply_sign_constraints, build_summary


def test_apply_sign_constraints_zeroes_wrong_sign_weights() -> None:
    weights = {
        "logic_prototype_proxy": -1.0,
        "logic_fragile_bridge_proxy": 2.0,
        "syntax_constraint_conflict_proxy": -3.0,
        "logic_control_proxy": 4.0,
    }
    adjusted = apply_sign_constraints(
        weights,
        {
            "logic_prototype_proxy": "positive",
            "logic_fragile_bridge_proxy": "negative",
            "syntax_constraint_conflict_proxy": "positive",
            "logic_control_proxy": "negative",
        },
    )
    assert adjusted["logic_prototype_proxy"] == 0.0
    assert adjusted["logic_fragile_bridge_proxy"] == 0.0
    assert adjusted["syntax_constraint_conflict_proxy"] == 0.0
    assert adjusted["logic_control_proxy"] == 0.0


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
        "logic_control_proxy": 0.8,
        "syntax_control_proxy": 0.9,
        "union_joint_adv": 0.2,
        "union_synergy_joint": 0.1,
        "strict_positive_synergy": 1.0,
    }
    summary = build_summary([row, dict(row, union_joint_adv=0.3, union_synergy_joint=0.0, strict_positive_synergy=0.0)])
    assert summary["record_type"] == "stage56_constrained_sample_regression_summary"
    assert len(summary["fits"]) == 3
