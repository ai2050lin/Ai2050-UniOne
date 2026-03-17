from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_synergy_conflict_dissection import (  # noqa: E402
    partition_union_indices,
    select_target_pair,
)


def test_select_target_pair_prefers_negative_synergy() -> None:
    rows = [
        {"category": "fruit", "instance_term": "apple", "union_synergy_joint": -0.03, "union_joint_adv": -0.01},
        {"category": "fruit", "instance_term": "apple", "union_synergy_joint": -0.01, "union_joint_adv": 0.02},
    ]
    target = select_target_pair(rows, category="fruit", instance_term="apple", prefer_conflict=True)
    assert float(target["union_synergy_joint"]) == -0.03


def test_partition_union_indices_handles_overlap() -> None:
    result = partition_union_indices([1, 2, 3], [3, 4, 5])
    assert result["prototype_only"] == [1, 2]
    assert result["instance_only"] == [4, 5]
    assert result["overlap"] == [3]
    assert result["union"] == [1, 2, 3, 4, 5]
