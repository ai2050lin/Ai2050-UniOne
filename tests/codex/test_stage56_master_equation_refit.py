from __future__ import annotations

from stage56_master_equation_refit import join_rows, build_summary


def test_join_rows_merges_new_static_window_style_terms() -> None:
    design_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "frontier_dynamic_proxy": 0.5,
            "logic_prototype_proxy": 0.4,
            "logic_fragile_bridge_proxy": 0.1,
            "syntax_constraint_conflict_proxy": 0.2,
            "logic_control_proxy": -0.3,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        }
    ]
    static_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "identity_margin_direct": 0.7,
        }
    ]
    window_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "generated_dominance_mean": 0.8,
        }
    ]
    style_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "style_alignment": 0.6,
            "style_midfield": 0.4,
        }
    ]
    rows = join_rows(design_rows, static_rows, window_rows, style_rows)
    assert len(rows) == 1
    assert rows[0]["identity_margin_term"] == 0.7
    assert rows[0]["window_dominance_term"] == 0.8
    assert rows[0]["style_alignment_term"] == 0.6


def test_build_summary_returns_refit_metadata() -> None:
    row = {
        "identity_margin_term": 0.7,
        "frontier_term": 0.5,
        "logic_prototype_term": 0.4,
        "logic_fragile_bridge_term": 0.1,
        "syntax_constraint_conflict_term": 0.2,
        "window_dominance_term": 0.8,
        "style_alignment_term": 0.6,
        "style_midfield_term": 0.4,
        "logic_control_term": -0.3,
        "union_joint_adv": 0.2,
        "union_synergy_joint": 0.1,
        "strict_positive_synergy": 1.0,
    }
    summary = build_summary([row, dict(row, union_joint_adv=0.1, union_synergy_joint=0.0, strict_positive_synergy=0.0)])
    assert summary["record_type"] == "stage56_master_equation_refit_summary"
    assert len(summary["fits"]) == 3
    assert "equation_text" in summary
