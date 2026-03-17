from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage6_prototype_instance_decomposition import (  # noqa: E402
    is_strict_positive_synergy,
    paired_categories,
    top_rows_by_category,
    unique_union,
)


def test_top_rows_by_category_keeps_best_per_category():
    rows = [
        {"item": {"category": "animal", "term": "cat"}, "full_joint_adv_score": 0.2, "full_strict_joint_adv_score": 0.2},
        {"item": {"category": "animal", "term": "dog"}, "full_joint_adv_score": 0.5, "full_strict_joint_adv_score": 0.1},
        {"item": {"category": "tech", "term": "data"}, "full_joint_adv_score": 0.1},
    ]
    out = top_rows_by_category(rows, per_category_limit=1)
    assert out["animal"][0]["full_joint_adv_score"] == 0.2
    assert out["tech"][0]["full_joint_adv_score"] == 0.1


def test_top_rows_by_category_dedupes_repeated_terms_within_category():
    rows = [
        {"item": {"category": "human", "term": "teacher"}, "full_joint_adv_score": 0.4, "full_strict_joint_adv_score": 0.4},
        {"item": {"category": "human", "term": "teacher"}, "full_joint_adv_score": 0.3, "full_strict_joint_adv_score": 0.3},
        {"item": {"category": "human", "term": "doctor"}, "full_joint_adv_score": 0.2, "full_strict_joint_adv_score": 0.2},
    ]
    out = top_rows_by_category(rows, per_category_limit=2)
    assert [row["item"]["term"] for row in out["human"]] == ["teacher", "doctor"]


def test_paired_categories_uses_intersection():
    left = {"animal": [{}], "tech": [{}]}
    right = {"animal": [{}], "human": [{}]}
    assert paired_categories(left, right) == ["animal"]


def test_unique_union_preserves_order_and_dedupes():
    assert unique_union([3, 2, 1], [2, 4, 1, 5]) == [3, 2, 1, 4, 5]


def test_is_strict_positive_synergy_requires_union_to_beat_both_single_paths():
    row = {
        "proto_joint_adv": 0.01,
        "instance_joint_adv": 0.02,
        "union_joint_adv": 0.03,
        "union_synergy_joint": 0.001,
    }
    assert is_strict_positive_synergy(row) is True
    row["union_joint_adv"] = 0.015
    assert is_strict_positive_synergy(row) is False
