from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_generation_gate_multimodel_compare import (  # noqa: E402
    aggregate_axis_block,
    aggregate_model_block,
    build_field_consensus,
    filter_rows,
)


def make_row(model_id: str, category: str, p: float, i: float, b: float, x: float, m: float):
    deltas = {
        "prototype_field_proxy": p,
        "instance_field_proxy": i,
        "bridge_field_proxy": b,
        "conflict_field_proxy": x,
        "mismatch_field_proxy": m,
    }
    axes = {
        axis: {
            "deltas": dict(deltas),
            "directions": {
                key: ("positive" if value > 0 else "negative" if value < 0 else "neutral")
                for key, value in deltas.items()
            },
        }
        for axis in ("style", "logic", "syntax")
    }
    return {
        "model_id": model_id,
        "group_label": f"{model_id}_{category}",
        "category": category,
        "axis_gate_summary": {"axes": axes},
    }


def test_aggregate_axis_block_picks_largest_magnitude_field() -> None:
    rows = [
        make_row("m1", "fruit", 0.1, 0.0, 0.0, 0.2, 0.3),
        make_row("m1", "food", 0.0, 0.0, 0.0, 0.2, 0.1),
    ]
    block = aggregate_axis_block(rows, "logic")
    assert block["case_count"] == 2
    assert abs(block["mean_deltas"]["conflict_field_proxy"] - 0.2) < 1e-9
    assert block["dominant_field"] == "mismatch_field_proxy"
    assert block["direction_signature"]["prototype_field_proxy"] == "positive"


def test_filter_rows_applies_requested_dimensions() -> None:
    rows = [
        make_row("m1", "fruit", 0, 0, 0, 0, 0),
        make_row("m2", "food", 0, 0, 0, 0, 0),
    ]
    rows[0]["group_label"] = "g1"
    rows[1]["group_label"] = "g2"
    out = filter_rows(rows, model_ids=["m1"], group_labels=["g1"], categories=["fruit"])
    assert len(out) == 1
    assert out[0]["model_id"] == "m1"


def test_build_field_consensus_requires_same_non_neutral_sign() -> None:
    summary = {
        "per_model": {
            "m1": aggregate_model_block([make_row("m1", "fruit", 0.1, 0.0, 0.0, 0.2, 0.3)]),
            "m2": aggregate_model_block([make_row("m2", "fruit", 0.2, 0.0, 0.0, 0.1, 0.4)]),
        }
    }
    consensus = build_field_consensus(summary)
    assert consensus["style"]["prototype_field_proxy"]["consensus"] == "positive"
    assert consensus["style"]["mismatch_field_proxy"]["consensus"] == "positive"

    summary = {
        "per_model": {
            "m1": aggregate_model_block([make_row("m1", "fruit", 0.1, 0.0, 0.0, 0.2, 0.3)]),
            "m2": aggregate_model_block([make_row("m2", "fruit", -0.2, 0.0, 0.0, 0.1, 0.4)]),
        }
    }
    consensus = build_field_consensus(summary)
    assert consensus["style"]["prototype_field_proxy"]["consensus"] == "mixed"
