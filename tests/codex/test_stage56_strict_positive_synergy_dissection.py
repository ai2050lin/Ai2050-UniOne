from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_strict_positive_synergy_dissection import (  # noqa: E402
    find_target_pair,
    partition_union_indices,
)


def test_find_target_pair_prefers_strict_positive_match():
    rows = [
        {
            "category": "human",
            "instance_term": "teacher",
            "union_joint_adv": 0.02,
            "strict_positive_synergy": False,
        },
        {
            "category": "human",
            "instance_term": "teacher",
            "union_joint_adv": 0.01,
            "strict_positive_synergy": True,
        },
    ]
    row = find_target_pair(rows, category="human", instance_term="teacher", require_strict_positive=True)
    assert row["strict_positive_synergy"] is True


def test_partition_union_indices_separates_proto_instance_and_overlap():
    out = partition_union_indices([1, 2, 3, 5], [3, 4, 5, 6])
    assert out["prototype_only"] == [1, 2]
    assert out["instance_only"] == [4, 6]
    assert out["overlap"] == [3, 5]
    assert out["union"] == [1, 2, 3, 5, 4, 6]
