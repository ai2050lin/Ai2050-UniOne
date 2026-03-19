from __future__ import annotations

from stage56_sign_stability_runner import build_summary


def test_build_summary_reports_stable_features() -> None:
    rows = [
        {
            "atlas_static_proxy": 0.6,
            "offset_static_proxy": 0.2,
            "frontier_dynamic_proxy": 0.5,
            "logic_prototype_proxy": 0.1,
            "logic_fragile_bridge_proxy": 0.01,
            "syntax_constraint_conflict_proxy": 0.2,
            "style_control_proxy": 0.1,
            "logic_control_proxy": 0.3,
            "syntax_control_proxy": 0.4,
            "union_joint_adv": 0.3,
            "union_synergy_joint": 0.2,
            "strict_positive_synergy": 1.0,
        },
        {
            "atlas_static_proxy": 0.4,
            "offset_static_proxy": 0.3,
            "frontier_dynamic_proxy": 0.4,
            "logic_prototype_proxy": 0.05,
            "logic_fragile_bridge_proxy": 0.02,
            "syntax_constraint_conflict_proxy": 0.1,
            "style_control_proxy": 0.2,
            "logic_control_proxy": 0.2,
            "syntax_control_proxy": 0.3,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 0.0,
        },
    ]
    summary = build_summary(rows)
    assert summary["record_type"] == "stage56_sign_stability_runner_summary"
    assert "sign_matrix" in summary
