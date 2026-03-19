from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_syntax_conflict_internal_dive import build_summary, join_rows  # noqa: E402


def make_pair(model_id: str, category: str, proto: str, inst: str, syntax_x: float, synergy: float, joint_adv: float) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "prototype_term": proto,
        "instance_term": inst,
        "strict_positive_synergy": synergy > 0 and joint_adv > 0,
        "union_synergy_joint": synergy,
        "union_joint_adv": joint_adv,
        "axes": {
            "syntax": {"conflict_field_proxy": syntax_x},
        },
    }


def make_case(model_id: str, category: str, proto: str, inst: str, hidden: str, mlp: str, head: str) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "prototype_term": proto,
        "instance_term": inst,
        "axis_internal_summary": {
            "syntax": {
                "mean_hidden_shift_profile": [0.1, 0.9, 0.2],
                "mean_mlp_layer_delta_profile": [0.2, 0.1, 0.8],
                "dominant_hidden_layer": hidden,
                "dominant_mlp_layer": mlp,
                "dominant_attention_head": head,
            }
        },
    }


def test_join_rows_splits_constraint_and_destructive_conflict() -> None:
    joined = join_rows(
        [
            make_pair("m1", "tech", "p", "i1", 0.3, 0.2, 0.1),
            make_pair("m1", "tech", "p", "i2", 0.5, -0.1, 0.1),
        ],
        [
            make_case("m1", "tech", "p", "i1", "layer_10", "layer_12", "layer_3_head_1"),
            make_case("m1", "tech", "p", "i2", "layer_20", "layer_22", "layer_7_head_3"),
        ],
    )
    assert len(joined) == 2
    assert joined[0]["constraint_weight"] == 0.3
    assert joined[0]["destructive_weight"] == 0.0
    assert joined[1]["constraint_weight"] == 0.0
    assert joined[1]["destructive_weight"] == 0.5


def test_build_summary_keeps_modes_for_two_subsets() -> None:
    joined = join_rows(
        [
            make_pair("m1", "tech", "p", "i1", 0.3, 0.2, 0.1),
            make_pair("m2", "human", "p", "i2", 0.5, -0.2, 0.1),
        ],
        [
            make_case("m1", "tech", "p", "i1", "layer_10", "layer_12", "layer_3_head_1"),
            make_case("m2", "human", "p", "i2", "layer_20", "layer_22", "layer_7_head_3"),
        ],
    )
    summary = build_summary(joined)
    assert summary["constraint_conflict"]["overall"]["dominant_hidden_layer_mode"] == "layer_10"
    assert summary["destructive_conflict"]["overall"]["dominant_hidden_layer_mode"] == "layer_20"
