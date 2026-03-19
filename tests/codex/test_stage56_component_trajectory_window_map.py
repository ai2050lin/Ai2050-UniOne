from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_component_trajectory_window_map import build_summary, join_rows  # noqa: E402


def make_component(component_label: str, model_id: str, category: str, proto: str, inst: str, weight: float) -> dict:
    return {
        "component_label": component_label,
        "model_id": model_id,
        "category": category,
        "prototype_term": proto,
        "instance_term": inst,
        "weight": weight,
        "union_joint_adv": 0.2,
        "union_synergy_joint": 0.1,
    }


def make_trajectory(axis: str, model_id: str, category: str, proto: str, inst: str, hidden_tail: str, mlp_tail: str) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "prototype_term": proto,
        "instance_term": inst,
        "axis": axis,
        "dominant_hidden_tail_position": hidden_tail,
        "dominant_mlp_tail_position": mlp_tail,
        "hidden_late_focus": 0.4,
        "mlp_late_focus": 0.5,
        "hidden_total": 10.0,
        "mlp_total": 1.0,
        "tail_position_labels": ["tail_pos_-3", "tail_pos_-2", "tail_pos_-1"],
        "mean_hidden_token_profile": [0.1, 0.8, 0.2],
        "mean_mlp_token_profile": [0.2, 0.1, 0.7],
    }


def test_join_rows_matches_component_label_to_axis() -> None:
    joined = join_rows(
        [
            make_component("logic_prototype", "m1", "tech", "proto", "inst", 0.5),
            make_component("syntax_constraint_conflict", "m1", "tech", "proto", "inst", 0.2),
        ],
        [
            make_trajectory("logic", "m1", "tech", "proto", "inst", "tail_pos_-2", "tail_pos_-1"),
            make_trajectory("syntax", "m1", "tech", "proto", "inst", "tail_pos_-3", "tail_pos_-1"),
        ],
    )
    assert len(joined) == 2
    assert {row["axis"] for row in joined} == {"logic", "syntax"}


def test_build_summary_keeps_tail_modes_per_component() -> None:
    joined = join_rows(
        [
            make_component("logic_prototype", "m1", "tech", "proto", "inst1", 0.6),
            make_component("logic_prototype", "m2", "human", "proto", "inst2", 0.1),
            make_component("logic_fragile_bridge", "m3", "weather", "proto", "inst3", 0.4),
        ],
        [
            make_trajectory("logic", "m1", "tech", "proto", "inst1", "tail_pos_-2", "tail_pos_-1"),
            make_trajectory("logic", "m2", "human", "proto", "inst2", "tail_pos_-3", "tail_pos_-1"),
            make_trajectory("logic", "m3", "weather", "proto", "inst3", "tail_pos_-3", "tail_pos_-2"),
        ],
    )
    summary = build_summary(joined)
    logic_proto = summary["per_component"]["logic_prototype"]["overall"]
    logic_fragile = summary["per_component"]["logic_fragile_bridge"]["overall"]
    assert logic_proto["dominant_hidden_tail_position_mode"] == "tail_pos_-2"
    assert logic_proto["peak_hidden_tail_position_from_profile"] == "tail_pos_-2"
    assert logic_fragile["dominant_hidden_tail_position_mode"] == "tail_pos_-3"
