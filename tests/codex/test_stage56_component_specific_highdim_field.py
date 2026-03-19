from __future__ import annotations

from stage56_component_specific_highdim_field import build_summary, join_rows


def _pair_row() -> dict:
    return {
        "model_id": "m",
        "model_label": "M",
        "category": "fruit",
        "prototype_term": "apple",
        "instance_term": "pear",
        "strict_positive_synergy": True,
        "union_joint_adv": 0.2,
        "union_synergy_joint": 0.1,
        "axes": {
            "logic": {
                "pair_compaction_curve": [0.1, 0.2, 0.3, 0.4, 0.5],
                "pair_coverage_curve": [0.5, 0.6, 0.7, 0.8, 0.9],
            },
            "syntax": {
                "pair_compaction_curve": [0.2, 0.3, 0.4, 0.5, 0.6],
                "pair_coverage_curve": [0.6, 0.7, 0.8, 0.9, 1.0],
            },
        },
    }


def _internal_row(component_label: str, axis: str, weight: float) -> dict:
    return {
        "component_label": component_label,
        "axis": axis,
        "model_id": "m",
        "category": "fruit",
        "prototype_term": "apple",
        "instance_term": "pear",
        "weight": weight,
        "hidden_profile": [0.0, 0.1, 0.5, 0.2],
        "mlp_profile": [0.0, 0.2, 0.4, 0.4],
    }


def _trajectory_row(component_label: str) -> dict:
    return {
        "component_label": component_label,
        "model_id": "m",
        "category": "fruit",
        "prototype_term": "apple",
        "instance_term": "pear",
        "hidden_token_profile": [0.0, 0.3, 0.7],
        "mlp_token_profile": [0.0, 0.4, 0.6],
    }


def test_join_rows_builds_component_specific_energies() -> None:
    joined = join_rows(
        [_pair_row()],
        [
            _internal_row("logic_prototype", "logic", 0.5),
            _internal_row("logic_fragile_bridge", "logic", 0.3),
            _internal_row("syntax_constraint_conflict", "syntax", 0.4),
        ],
        [
            _trajectory_row("logic_prototype"),
            _trajectory_row("logic_fragile_bridge"),
            _trajectory_row("syntax_constraint_conflict"),
        ],
    )
    assert len(joined) == 3
    assert joined[0]["tensor_shape"] == [4, 3]
    assert joined[0]["layer_window_hidden_energy"] >= 0.0


def test_build_summary_collects_component_stats() -> None:
    rows = join_rows(
        [_pair_row()],
        [
            _internal_row("logic_prototype", "logic", 0.5),
            _internal_row("logic_fragile_bridge", "logic", 0.3),
            _internal_row("syntax_constraint_conflict", "syntax", 0.4),
        ],
        [
            _trajectory_row("logic_prototype"),
            _trajectory_row("logic_fragile_bridge"),
            _trajectory_row("syntax_constraint_conflict"),
        ],
    )
    summary = build_summary(rows)
    assert summary["joined_row_count"] == 3
    assert "logic_prototype" in summary["component_labels"]
    assert summary["top_abs_correlations"]
