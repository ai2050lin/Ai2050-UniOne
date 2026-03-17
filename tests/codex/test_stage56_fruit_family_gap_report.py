from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_fruit_family_gap_report import (  # noqa: E402
    aggregate_category_rows,
    analyze_model_gap,
    build_payload,
)


def test_aggregate_category_rows_merges_models() -> None:
    rows = [
        {"category": "fruit", "pair_count": 2, "strict_positive_pair_count": 0, "mean_union_joint_adv": 0.01, "mean_union_synergy_joint": -0.02},
        {"category": "fruit", "pair_count": 3, "strict_positive_pair_count": 1, "mean_union_joint_adv": 0.02, "mean_union_synergy_joint": -0.01},
    ]
    merged = aggregate_category_rows(rows, "fruit")
    assert merged["pair_count"] == 5
    assert merged["strict_positive_pair_count"] == 1
    assert merged["strict_positive_pair_ratio"] == 0.2


def test_build_payload_marks_stage3_selection_gap() -> None:
    fruit_discovery = {
        "category": "fruit",
        "pair_count": 5,
        "strict_positive_pair_count": 1,
        "strict_positive_pair_ratio": 0.2,
        "mean_union_joint_adv": 0.01,
        "mean_union_synergy_joint": -0.01,
    }
    apple_multiaxis = {
        "concept_axis": {
            "micro_to_meso_jaccard_mean": 0.02,
            "meso_to_macro_jaccard_mean": 0.37,
            "shared_base_ratio_mean": 0.03,
        }
    }
    qwen_gap = analyze_model_gap(
        {"selected_categories": ["tech", "human"]},
        {"mean_candidate_full_strict_joint_adv": 0.0},
        {"mean_candidate_full_strict_joint_adv": -0.04},
        {"mean_union_joint_adv": 0.02, "mean_union_synergy_joint": -0.004, "strict_positive_synergy_pair_count": 1},
        "fruit",
    )
    deepseek_gap = analyze_model_gap(
        {"selected_categories": ["animal", "vehicle"]},
        {"mean_candidate_full_strict_joint_adv": -0.05},
        {"mean_candidate_full_strict_joint_adv": -0.05},
        {"mean_union_joint_adv": -0.002, "mean_union_synergy_joint": -0.001, "strict_positive_synergy_pair_count": 0},
        "fruit",
    )
    payload = build_payload(fruit_discovery, apple_multiaxis, qwen_gap, deepseek_gap)
    assert qwen_gap["selected_in_stage3"] is False
    assert deepseek_gap["selected_in_stage3"] is False
    assert any("stage3" in line for line in payload["failure_modes"])
