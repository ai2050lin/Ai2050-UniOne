from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_large_relation_atlas import classify_relation_record, summarize  # noqa: E402


def test_gender_swap_is_local_linear() -> None:
    row = classify_relation_record(
        {"family": "gender_role_swap", "kind": "quadruplet", "word_class": "noun", "items": ["king", "man", "woman", "queen"], "category": "human"},
        axis_specificity=0.63,
        hierarchy_gain=0.35,
        decoupling=0.68,
        protocol_gap=0.14,
        category_metrics={"human": {"strict_positive_pair_ratio": 0.0, "mean_union_joint_adv": 0.005, "mean_union_synergy_joint": -0.02}},
        abstract_penalty=0.08,
    )
    assert row["interpretation"] == "local_linear"
    assert row["local_linear_score"] > row["path_bundle_score"]


def test_category_instance_is_path_bundle() -> None:
    row = classify_relation_record(
        {"family": "category_instance_triplet", "kind": "triplet", "word_class": "noun", "items": ["fruit", "apple", "banana"], "category": "fruit"},
        axis_specificity=0.63,
        hierarchy_gain=0.35,
        decoupling=0.68,
        protocol_gap=0.14,
        category_metrics={"fruit": {"strict_positive_pair_ratio": 0.5, "mean_union_joint_adv": 0.30, "mean_union_synergy_joint": 0.04}},
        abstract_penalty=0.08,
    )
    assert row["interpretation"] == "path_bundle"


def test_summary_counts_interpretations() -> None:
    summary = summarize(
        [
            {"interpretation": "local_linear", "word_class": "noun", "family": "gender_role_swap", "local_linear_score": 0.8, "path_bundle_score": 0.3},
            {"interpretation": "path_bundle", "word_class": "verb", "family": "verb_process_chain", "local_linear_score": 0.3, "path_bundle_score": 0.8},
        ]
    )
    assert summary["group_count"] == 2
    assert summary["counts_by_interpretation"]["local_linear"] == 1
    assert summary["counts_by_interpretation"]["path_bundle"] == 1
