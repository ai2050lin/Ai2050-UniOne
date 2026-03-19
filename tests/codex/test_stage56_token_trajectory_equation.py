from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_token_trajectory_equation import (  # noqa: E402
    aggregate_axis_rows,
    build_axis_stage6_link,
    build_equation_summary,
    build_trace_delta,
    dominant_tail_position,
    select_representative_cases,
    summarize_case_axis,
    tail_align,
)


def test_tail_align_left_pads_short_profiles() -> None:
    profile = tail_align([1.0, 2.0], 4).tolist()
    assert profile == [0.0, 0.0, 1.0, 2.0]


def test_build_trace_delta_extracts_tail_positions_and_layers() -> None:
    control = {
        "input_ids": [1, 2, 3],
        "hidden_seq": [[[1.0], [1.0], [1.0]], [[2.0], [2.0], [2.0]]],
        "gate_seq": [[[1.0], [1.0], [1.0]], [[2.0], [2.0], [2.0]]],
        "attention_seq": [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]],
    }
    variant = {
        "input_ids": [1, 2, 3],
        "hidden_seq": [[[1.0], [3.0], [5.0]], [[2.0], [2.0], [4.0]]],
        "gate_seq": [[[1.0], [2.0], [3.0]], [[2.0], [3.0], [4.0]]],
        "attention_seq": [[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]],
    }
    row = build_trace_delta(control=control, variant=variant, tail_tokens=4)
    assert row["dominant_hidden_tail_position"] == "tail_pos_-1"
    assert row["dominant_mlp_layer"] == "layer_1"


def test_aggregate_axis_rows_and_equation_summary_keep_token_peaks() -> None:
    axis_block = aggregate_axis_rows(
        [
            {
                "hidden_token_profile": [0.0, 0.2, 0.8],
                "mlp_token_profile": [0.0, 0.3, 0.7],
                "hidden_layer_profile": [0.1, 0.9],
                "mlp_layer_profile": [0.2, 0.8],
                "attention_head_profile": [[0.1, 0.2], [0.3, 0.4]],
            },
            {
                "hidden_token_profile": [0.0, 0.1, 0.9],
                "mlp_token_profile": [0.0, 0.2, 0.8],
                "hidden_layer_profile": [0.2, 0.8],
                "mlp_layer_profile": [0.1, 0.9],
                "attention_head_profile": [[0.0, 0.2], [0.1, 0.5]],
            },
        ],
        tail_tokens=3,
    )
    assert axis_block["dominant_hidden_tail_position"] == dominant_tail_position([0.0, 0.15, 0.85])
    summary = build_equation_summary(
        {
            "tail_tokens": 3,
            "per_model": {
                "demo": {
                    "per_axis": {
                        "syntax": axis_block,
                    }
                }
            },
        },
        {
            "equations": {
                "closure_equation": {
                    "coefficients": {
                        "logic_P": 0.4,
                        "syntax_CX": 0.2,
                    }
                }
            }
        },
    )
    assert summary["per_model_equations"]["demo"]["axes"]["syntax"]["dominant_hidden_layer"] == "layer_1"


def test_summarize_case_axis_and_stage6_link_keep_axis_correlations() -> None:
    case = {
        "model_id": "demo",
        "category": "fruit",
        "prototype_term": "fruit",
        "instance_term": "apple",
        "case_role": "weak_bridge_positive",
        "stage6_reference": {
            "proto_joint_adv": 0.3,
            "instance_joint_adv": 0.1,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.05,
        },
    }
    row = summarize_case_axis(
        case=case,
        axis="logic",
        axis_rows=[
            {
                "hidden_token_profile": [0.0, 0.2, 0.8],
                "mlp_token_profile": [0.0, 0.3, 0.7],
                "hidden_layer_profile": [0.1, 0.9],
                "mlp_layer_profile": [0.2, 0.8],
                "attention_head_profile": [[0.1, 0.2], [0.3, 0.4]],
            }
        ],
        tail_tokens=3,
    )
    assert row["hidden_total"] > 0.0
    link = build_axis_stage6_link(
        [
            row,
            {
                **row,
                "case_key": "demo|fruit|pear",
                "hidden_total": row["hidden_total"] * 1.2,
                "mlp_total": row["mlp_total"] * 1.2,
                "hidden_late_focus": row["hidden_late_focus"] * 0.9,
                "mlp_late_focus": row["mlp_late_focus"] * 0.9,
                "proto_joint_adv": 0.4,
                "instance_joint_adv": 0.12,
                "union_joint_adv": 0.25,
                "union_synergy_joint": 0.08,
            },
        ]
    )
    assert link["logic"]["case_count"] == 2
    assert link["logic"]["corr_hidden_total_to_proto_joint_adv"] > 0.0


def test_select_representative_cases_allows_all_cases_when_limit_non_positive() -> None:
    rows = [
        {"model_id": "m1", "case_role": "weak_bridge_positive", "category": "a", "instance_term": "x"},
        {"model_id": "m1", "case_role": "bridge_dominant", "category": "b", "instance_term": "y"},
        {"model_id": "m2", "case_role": "strict_positive_pair", "category": "c", "instance_term": "z"},
    ]
    selected = select_representative_cases(rows, max_cases_per_model=0)
    assert len(selected) == 3


def test_summarize_case_axis_case_key_keeps_prototype_and_instance() -> None:
    case = {
        "model_id": "demo",
        "category": "fruit",
        "prototype_term": "melon",
        "instance_term": "papaya",
        "case_role": "weak_bridge_positive",
        "stage6_reference": {
            "proto_joint_adv": 0.2,
            "instance_joint_adv": 0.1,
            "union_joint_adv": 0.3,
            "union_synergy_joint": 0.04,
        },
    }
    row = summarize_case_axis(
        case=case,
        axis="syntax",
        axis_rows=[
            {
                "hidden_token_profile": [0.0, 0.5],
                "mlp_token_profile": [0.0, 0.4],
                "hidden_layer_profile": [0.2, 0.8],
                "mlp_layer_profile": [0.1, 0.9],
                "attention_head_profile": [[0.1, 0.2]],
            }
        ],
        tail_tokens=2,
    )
    assert row["case_key"] == "demo|fruit|melon|papaya"
