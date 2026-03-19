from __future__ import annotations

from stage56_complete_highdim_field import build_summary, join_rows


def _component_row(component_label: str, axis: str) -> dict:
    return {
        "model_id": "m",
        "model_label": "M",
        "category": "fruit",
        "prototype_term": "apple",
        "instance_term": "pear",
        "axis": axis,
        "component_label": component_label,
        "strict_positive_synergy": True,
        "union_joint_adv": 0.2,
        "union_synergy_joint": 0.1,
        "weight": 0.5,
        "preferred_density": 0.3,
        "hidden_layer_center": 10.0,
        "mlp_layer_center": 11.0,
        "hidden_window_center": 8.0,
        "mlp_window_center": 9.0,
        "layer_window_hidden_energy": 0.02,
        "layer_window_mlp_energy": 0.03,
        "layer_window_cross_energy": 0.025,
    }


def _natural_row(axis: str) -> dict:
    return {
        "model_id": "m",
        "category": "fruit",
        "prototype_term": "apple",
        "instance_term": "pear",
        "axis": axis,
        "hidden_generated_share": 0.7,
        "mlp_generated_share": 0.6,
        "hidden_prompt_share": 0.3,
        "mlp_prompt_share": 0.4,
        "hidden_generated_sum": 7.0,
        "hidden_prompt_sum": 3.0,
        "mlp_generated_sum": 6.0,
        "mlp_prompt_sum": 4.0,
    }


def test_join_rows_adds_generation_side_features() -> None:
    rows = join_rows(
        [
            _component_row("logic_prototype", "logic"),
            _component_row("syntax_constraint_conflict", "syntax"),
        ],
        [
            _natural_row("logic"),
            _natural_row("syntax"),
        ],
    )
    assert len(rows) == 2
    assert rows[0]["complete_generated_energy"] > rows[0]["complete_prompt_energy"]
    assert rows[0]["generated_dominance_score"] == 0.65


def test_build_summary_collects_complete_field_stats() -> None:
    rows = join_rows(
        [
            _component_row("logic_prototype", "logic"),
            _component_row("logic_fragile_bridge", "logic"),
            _component_row("syntax_constraint_conflict", "syntax"),
        ],
        [
            _natural_row("logic"),
            _natural_row("logic"),
            _natural_row("syntax"),
        ],
    )
    summary = build_summary(rows)
    assert summary["joined_row_count"] == 3
    assert "logic_prototype" in summary["component_labels"]
    assert summary["top_abs_correlations"]
