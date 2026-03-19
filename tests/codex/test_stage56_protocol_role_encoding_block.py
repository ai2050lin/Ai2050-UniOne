from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_protocol_role_encoding_block import (  # noqa: E402
    aggregate_stage6_category,
    build_summary,
)


def make_row(category: str, proto: float, inst: float, union: float, synergy: float, overlap: int = 4, union_count: int = 8, strict: bool = False) -> dict:
    return {
        "category": category,
        "prototype_term": "p",
        "instance_term": "i",
        "proto_joint_adv": proto,
        "instance_joint_adv": inst,
        "union_joint_adv": union,
        "union_synergy_joint": synergy,
        "overlap_neuron_count": overlap,
        "union_neuron_count": union_count,
        "strict_positive_synergy": strict,
        "model_id": "m1",
    }


def test_aggregate_stage6_category_detects_protocol_role_pattern() -> None:
    summary = aggregate_stage6_category(
        [
            make_row("tech", 0.01, 0.05, 0.02, -0.02),
            make_row("tech", -0.01, 0.04, 0.01, -0.03),
        ]
    )
    assert summary["encoding_class"] == "protocol_role_dominant"
    assert summary["mean_instance_proto_gap"] > 0.0
    assert summary["mean_union_synergy_joint"] < 0.0


def test_build_summary_compares_focus_and_anchor_scores() -> None:
    stage6_rows = [
        make_row("tech", 0.01, 0.05, 0.02, -0.02),
        make_row("human", 0.0, 0.03, 0.01, -0.01),
        make_row("action", 0.01, 0.02, 0.03, 0.005, strict=True),
        make_row("fruit", 0.03, 0.01, 0.05, 0.02, strict=True),
        make_row("object", 0.02, 0.01, 0.04, 0.01, strict=True),
        make_row("animal", 0.01, 0.01, -0.01, -0.02),
    ]
    discovery_rows = [
        {"category": "tech", "model_tag": "m1", "top_instance_term": "protocol", "strict_positive_pair_ratio": 0.0, "mean_union_joint_adv": 0.02, "mean_union_synergy_joint": -0.02},
        {"category": "fruit", "model_tag": "m1", "top_instance_term": "kiwi", "strict_positive_pair_ratio": 1.0, "mean_union_joint_adv": 0.05, "mean_union_synergy_joint": 0.02},
    ]
    summary = build_summary(stage6_rows, discovery_rows, ["tech", "human", "action"], ["fruit", "object"])
    assert summary["focus_mean_protocol_role_pressure"] > summary["anchor_mean_protocol_role_pressure"]
    assert summary["discovery_support"]["tech"]["m1"]["top_instance_term"] == "protocol"
