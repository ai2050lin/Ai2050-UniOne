from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage6_prototype_instance_decomposition import (  # noqa: E402
    paired_categories,
    top_rows_by_category,
    unique_union,
)


def test_top_rows_by_category_keeps_best_per_category():
    rows = [
        {"item": {"category": "animal"}, "full_joint_adv_score": 0.2},
        {"item": {"category": "animal"}, "full_joint_adv_score": 0.5},
        {"item": {"category": "tech"}, "full_joint_adv_score": 0.1},
    ]
    out = top_rows_by_category(rows, per_category_limit=1)
    assert out["animal"][0]["full_joint_adv_score"] == 0.5
    assert out["tech"][0]["full_joint_adv_score"] == 0.1


def test_paired_categories_uses_intersection():
    left = {"animal": [{}], "tech": [{}]}
    right = {"animal": [{}], "human": [{}]}
    assert paired_categories(left, right) == ["animal"]


def test_unique_union_preserves_order_and_dedupes():
    assert unique_union([3, 2, 1], [2, 4, 1, 5]) == [3, 2, 1, 4, 5]
