from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_field_internal_subfield_map import build_summary, join_rows  # noqa: E402


def make_rewritten(
    model_id: str,
    category: str,
    proto: str,
    inst: str,
    logic_p: float,
    syntax_cx: float,
    logic_fb: float,
) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "prototype_term": proto,
        "instance_term": inst,
        "union_joint_adv": 0.2,
        "union_synergy_joint": 0.1,
        "strict_positive_synergy": True,
        "axes": {
            "logic": {"prototype_field_proxy": logic_p},
            "syntax": {"prototype_field_proxy": 0.0},
        },
        "rewritten_axes": {
            "logic": {"fragile_bridge": logic_fb},
            "syntax": {"constraint_conflict": syntax_cx},
        },
    }


def make_case(model_id: str, category: str, proto: str, inst: str, logic_hidden: str, syntax_hidden: str) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "prototype_term": proto,
        "instance_term": inst,
        "axis_internal_summary": {
            "logic": {
                "dominant_hidden_layer": logic_hidden,
                "dominant_mlp_layer": "layer_5",
                "dominant_attention_head": "layer_2_head_1",
                "mean_hidden_shift_profile": [0.1, 0.8, 0.2],
                "mean_mlp_layer_delta_profile": [0.2, 0.5, 0.1],
            },
            "syntax": {
                "dominant_hidden_layer": syntax_hidden,
                "dominant_mlp_layer": "layer_7",
                "dominant_attention_head": "layer_3_head_2",
                "mean_hidden_shift_profile": [0.2, 0.1, 0.9],
                "mean_mlp_layer_delta_profile": [0.1, 0.3, 0.7],
            },
        },
    }


def test_join_rows_keeps_three_component_channels() -> None:
    joined = join_rows(
        [
            make_rewritten("m1", "tech", "system", "protocol", logic_p=0.4, syntax_cx=0.2, logic_fb=0.1),
        ],
        [
            make_case("m1", "tech", "system", "protocol", logic_hidden="layer_10", syntax_hidden="layer_20"),
        ],
    )
    assert len(joined) == 3
    labels = {row["component_label"] for row in joined}
    assert labels == {"logic_prototype", "syntax_constraint_conflict", "logic_fragile_bridge"}


def test_build_summary_picks_modes_from_weighted_internal_support() -> None:
    joined = join_rows(
        [
            make_rewritten("m1", "tech", "system", "protocol", logic_p=0.6, syntax_cx=0.0, logic_fb=0.0),
            make_rewritten("m2", "human", "role", "teacher", logic_p=0.2, syntax_cx=0.5, logic_fb=0.4),
        ],
        [
            make_case("m1", "tech", "system", "protocol", logic_hidden="layer_10", syntax_hidden="layer_20"),
            make_case("m2", "human", "role", "teacher", logic_hidden="layer_30", syntax_hidden="layer_40"),
        ],
    )
    summary = build_summary(joined)
    logic_proto = summary["per_component"]["logic_prototype"]["overall"]
    syntax_cx = summary["per_component"]["syntax_constraint_conflict"]["overall"]
    logic_fb = summary["per_component"]["logic_fragile_bridge"]["overall"]
    assert logic_proto["dominant_hidden_layer_mode"] == "layer_10"
    assert syntax_cx["dominant_hidden_layer_mode"] == "layer_40"
    assert logic_fb["dominant_hidden_layer_mode"] == "layer_30"
