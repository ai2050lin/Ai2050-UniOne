from __future__ import annotations

from stage56_multicategory_strong_weak_taxonomy import dominant_structure


def test_dominant_structure_bridge_case() -> None:
    best_strong = {"metrics": {"joint_adv_mean": 0.01}}
    best_weak = {"metrics": {"joint_adv_mean": -0.01}}
    best_mixed = {"metrics": {"joint_adv_mean": 0.03}}
    assert dominant_structure("weak_bridge_positive", best_strong, best_weak, best_mixed) == "bridge_dominant"


def test_dominant_structure_strong_core_case() -> None:
    best_strong = {"metrics": {"joint_adv_mean": 0.04}}
    best_weak = {"metrics": {"joint_adv_mean": 0.01}}
    best_mixed = {"metrics": {"joint_adv_mean": 0.02}}
    assert dominant_structure("weak_drag_or_conflict", best_strong, best_weak, best_mixed) == "strong_core_dominant"
