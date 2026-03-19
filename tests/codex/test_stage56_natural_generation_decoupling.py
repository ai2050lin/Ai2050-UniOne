from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_natural_generation_decoupling import (  # noqa: E402
    attach_zone_metrics,
    build_summary,
    join_component_rows,
    zone_split,
)


def test_zone_split_uses_last_tokens_as_generated_zone() -> None:
    prompt_end, generated = zone_split(length=16, generated_token_count=8)
    assert prompt_end == 8
    assert generated == 8


def test_attach_zone_metrics_splits_prompt_and_generated_shares() -> None:
    row = {
        "model_id": "demo",
        "category": "fruit",
        "prototype_term": "papaya",
        "instance_term": "melon",
        "axis": "syntax",
        "generated_token_count": 2,
        "tail_position_labels": [f"tail_pos_-{idx}" for idx in range(4, 0, -1)],
        "mean_hidden_token_profile": [3.0, 1.0, 5.0, 7.0],
        "mean_mlp_token_profile": [0.5, 0.5, 2.0, 2.0],
    }
    enriched = attach_zone_metrics(row)
    assert enriched["prompt_window_token_count"] == 2
    assert enriched["generated_window_token_count"] == 2
    assert round(enriched["hidden_prompt_share"], 4) == 0.25
    assert round(enriched["hidden_generated_share"], 4) == 0.75
    assert enriched["dominant_hidden_zone"] == "generated"


def test_join_component_rows_matches_axis_to_natural_rows() -> None:
    component_rows = [
        {
            "component_label": "logic_prototype",
            "model_id": "demo",
            "category": "fruit",
            "prototype_term": "papaya",
            "instance_term": "melon",
            "weight": 0.2,
            "union_synergy_joint": 0.1,
            "union_joint_adv": 0.3,
        }
    ]
    natural_rows = [
        attach_zone_metrics(
            {
                "model_id": "demo",
                "category": "fruit",
                "prototype_term": "papaya",
                "instance_term": "melon",
                "axis": "logic",
                "generated_token_count": 2,
                "tail_position_labels": ["tail_pos_-4", "tail_pos_-3", "tail_pos_-2", "tail_pos_-1"],
                "mean_hidden_token_profile": [1.0, 1.0, 4.0, 4.0],
                "mean_mlp_token_profile": [0.1, 0.1, 0.4, 0.4],
            }
        )
    ]
    joined = join_component_rows(component_rows, natural_rows)
    assert len(joined) == 1
    assert joined[0]["axis"] == "logic"
    assert joined[0]["hidden_generated_sum"] > joined[0]["hidden_prompt_sum"]


def test_build_summary_keeps_axis_and_component_sections() -> None:
    natural_rows = [
        {
            "model_id": "demo",
            "category": "fruit",
            "prototype_term": "papaya",
            "instance_term": "melon",
            "axis": "syntax",
            "generated_token_count": 2,
            "tail_position_labels": ["tail_pos_-4", "tail_pos_-3", "tail_pos_-2", "tail_pos_-1"],
            "mean_hidden_token_profile": [5.0, 5.0, 1.0, 1.0],
            "mean_mlp_token_profile": [4.0, 4.0, 1.0, 1.0],
        }
    ]
    component_rows = [
        {
            "component_label": "syntax_constraint_conflict",
            "model_id": "demo",
            "category": "fruit",
            "prototype_term": "papaya",
            "instance_term": "melon",
            "weight": 0.2,
            "union_synergy_joint": 0.1,
            "union_joint_adv": 0.3,
        }
    ]
    summary = build_summary(natural_rows, component_rows)
    assert "syntax" in summary["per_axis"]
    assert "syntax_constraint_conflict" in summary["per_component"]
