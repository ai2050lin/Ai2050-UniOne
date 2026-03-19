from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_generation_gate_stage6_pair_link import (  # noqa: E402
    build_summary,
    join_gate_and_stage6_pairs,
)


def make_gate_row(model_id: str, category: str, proto: str, inst: str, style_b: float, logic_m: float) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "prototype_term": proto,
        "instance_term": inst,
        "group_label": f"{model_id}_{category}",
        "axis_gate_summary": {
            "axes": {
                "style": {
                    "deltas": {
                        "prototype_field_proxy": 0.0,
                        "instance_field_proxy": 0.0,
                        "bridge_field_proxy": style_b,
                        "conflict_field_proxy": 0.0,
                        "mismatch_field_proxy": 0.0,
                    }
                },
                "logic": {
                    "deltas": {
                        "prototype_field_proxy": 0.0,
                        "instance_field_proxy": 0.0,
                        "bridge_field_proxy": 0.0,
                        "conflict_field_proxy": 0.0,
                        "mismatch_field_proxy": logic_m,
                    }
                },
                "syntax": {
                    "deltas": {
                        "prototype_field_proxy": 0.0,
                        "instance_field_proxy": 0.0,
                        "bridge_field_proxy": 0.0,
                        "conflict_field_proxy": 0.0,
                        "mismatch_field_proxy": 0.0,
                    }
                },
            }
        },
    }


def make_stage6_row(model_id: str, category: str, proto: str, inst: str, union: float, synergy: float, strict: bool) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "prototype_term": proto,
        "instance_term": inst,
        "proto_joint_adv": union / 2.0,
        "instance_joint_adv": union / 3.0,
        "union_joint_adv": union,
        "union_synergy_joint": synergy,
        "strict_positive_synergy": strict,
    }


def test_join_gate_and_stage6_pairs_uses_exact_terms() -> None:
    gate_rows = [
        make_gate_row("m1", "fruit", "fruit", "apple", 0.2, 0.1),
        make_gate_row("m1", "fruit", "fruit", "pear", 0.3, 0.2),
    ]
    stage6_rows = {
        ("m1", "fruit", "fruit", "pear"): make_stage6_row("m1", "fruit", "fruit", "pear", 0.4, 0.1, True),
    }
    joined = join_gate_and_stage6_pairs(gate_rows, stage6_rows)
    assert len(joined) == 1
    assert joined[0]["instance_term"] == "pear"


def test_build_summary_detects_positive_and_negative_pair_associations() -> None:
    joined = [
        {
            "model_id": "m1",
            "category": "fruit",
            "prototype_term": "fruit",
            "instance_term": "apple",
            "strict_positive_synergy": True,
            "union_joint_adv": 0.3,
            "union_synergy_joint": 0.2,
            "proto_joint_adv": 0.1,
            "instance_joint_adv": 0.1,
            "axes": {
                "style": {"prototype_field_proxy": 0.0, "instance_field_proxy": 0.0, "bridge_field_proxy": 0.8, "conflict_field_proxy": 0.0, "mismatch_field_proxy": 0.0},
                "logic": {"prototype_field_proxy": 0.0, "instance_field_proxy": 0.0, "bridge_field_proxy": 0.0, "conflict_field_proxy": 0.0, "mismatch_field_proxy": -0.8},
                "syntax": {"prototype_field_proxy": 0.0, "instance_field_proxy": 0.0, "bridge_field_proxy": 0.0, "conflict_field_proxy": 0.0, "mismatch_field_proxy": 0.0},
            },
        },
        {
            "model_id": "m2",
            "category": "food",
            "prototype_term": "food",
            "instance_term": "bread",
            "strict_positive_synergy": False,
            "union_joint_adv": -0.2,
            "union_synergy_joint": -0.3,
            "proto_joint_adv": -0.1,
            "instance_joint_adv": -0.1,
            "axes": {
                "style": {"prototype_field_proxy": 0.0, "instance_field_proxy": 0.0, "bridge_field_proxy": -0.7, "conflict_field_proxy": 0.0, "mismatch_field_proxy": 0.0},
                "logic": {"prototype_field_proxy": 0.0, "instance_field_proxy": 0.0, "bridge_field_proxy": 0.0, "conflict_field_proxy": 0.0, "mismatch_field_proxy": 0.9},
                "syntax": {"prototype_field_proxy": 0.0, "instance_field_proxy": 0.0, "bridge_field_proxy": 0.0, "conflict_field_proxy": 0.0, "mismatch_field_proxy": 0.0},
            },
        },
    ]
    summary = build_summary(joined)
    assert summary["axis_target_stats"]["style"]["bridge_field_proxy"]["association_to_pair_synergy"] == "positive"
    assert summary["axis_target_stats"]["logic"]["mismatch_field_proxy"]["association_to_pair_synergy"] == "negative"
