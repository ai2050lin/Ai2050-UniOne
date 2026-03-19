from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_generation_gate_stage6_link import (  # noqa: E402
    aggregate_gate_by_model_category,
    build_summary,
    join_gate_and_stage6,
)


def make_gate_row(
    model_id: str,
    category: str,
    style_b: float,
    logic_m: float,
    syntax_x: float,
) -> dict:
    def axis_payload(p: float, i: float, b: float, x: float, m: float) -> dict:
        return {
            "deltas": {
                "prototype_field_proxy": p,
                "instance_field_proxy": i,
                "bridge_field_proxy": b,
                "conflict_field_proxy": x,
                "mismatch_field_proxy": m,
            }
        }

    return {
        "model_id": model_id,
        "category": category,
        "group_label": f"{model_id}_{category}",
        "axis_gate_summary": {
            "axes": {
                "style": axis_payload(0.0, 0.0, style_b, 0.0, 0.0),
                "logic": axis_payload(0.0, 0.0, 0.0, 0.0, logic_m),
                "syntax": axis_payload(0.0, 0.0, 0.0, syntax_x, 0.0),
            }
        },
    }


def make_stage6_row(
    model_id: str,
    category: str,
    union_joint_adv: float,
    union_synergy_joint: float,
    strict_ratio: float,
) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "pair_count": 2,
        "strict_positive_pair_count": 1 if strict_ratio > 0.0 else 0,
        "strict_positive_pair_ratio": strict_ratio,
        "mean_union_joint_adv": union_joint_adv,
        "mean_union_synergy_joint": union_synergy_joint,
        "mean_overlap_ratio": 0.5,
        "top_instance_term": "demo",
        "top_row_is_strict_positive": strict_ratio > 0.0,
    }


def test_join_gate_and_stage6_matches_model_category_keys() -> None:
    gate_rows = aggregate_gate_by_model_category(
        [
            make_gate_row("m1", "fruit", 0.3, 0.1, -0.2),
            make_gate_row("m1", "fruit", 0.1, 0.3, -0.1),
            make_gate_row("m2", "food", 0.2, 0.4, 0.0),
        ]
    )
    stage6_rows = {
        ("m1", "fruit"): make_stage6_row("m1", "fruit", 0.2, 0.1, 1.0),
        ("m2", "food"): make_stage6_row("m2", "food", -0.1, -0.2, 0.0),
    }
    joined = join_gate_and_stage6(gate_rows, stage6_rows)
    assert len(joined) == 2
    fruit = [row for row in joined if row["model_id"] == "m1" and row["category"] == "fruit"][0]
    assert abs(fruit["axes"]["style"]["bridge_field_proxy"] - 0.2) < 1e-9
    assert abs(fruit["axes"]["logic"]["mismatch_field_proxy"] - 0.2) < 1e-9


def test_build_summary_marks_positive_association_when_gate_tracks_synergy() -> None:
    gate_rows = aggregate_gate_by_model_category(
        [
            make_gate_row("m1", "fruit", 0.9, 0.8, -0.7),
            make_gate_row("m1", "food", -0.2, -0.1, 0.2),
            make_gate_row("m2", "fruit", 0.7, 0.9, -0.6),
            make_gate_row("m2", "food", -0.3, -0.2, 0.4),
        ]
    )
    stage6_rows = {
        ("m1", "fruit"): make_stage6_row("m1", "fruit", 0.4, 0.3, 1.0),
        ("m1", "food"): make_stage6_row("m1", "food", -0.2, -0.3, 0.0),
        ("m2", "fruit"): make_stage6_row("m2", "fruit", 0.5, 0.4, 1.0),
        ("m2", "food"): make_stage6_row("m2", "food", -0.1, -0.2, 0.0),
    }
    joined = join_gate_and_stage6(gate_rows, stage6_rows)
    summary = build_summary(joined)
    style_bridge = summary["axis_target_stats"]["style"]["bridge_field_proxy"]
    logic_mismatch = summary["axis_target_stats"]["logic"]["mismatch_field_proxy"]
    syntax_conflict = summary["axis_target_stats"]["syntax"]["conflict_field_proxy"]

    assert style_bridge["association_to_closure"] == "positive"
    assert logic_mismatch["association_to_closure"] == "positive"
    assert syntax_conflict["association_to_closure"] == "negative"
    assert summary["top_findings"]["top_synergy_associations"][0]["axis"] in {"style", "logic", "syntax"}
