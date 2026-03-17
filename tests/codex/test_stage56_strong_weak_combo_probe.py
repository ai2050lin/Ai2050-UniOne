from __future__ import annotations

from stage56_strong_weak_combo_probe import build_probe_subsets, split_strong_weak, unique_union


def test_unique_union_preserves_first_occurrence() -> None:
    assert unique_union([3, 1, 3], [2, 1, 4]) == [3, 1, 2, 4]


def test_split_strong_weak_uses_top_half_as_strong() -> None:
    rows = [
        {"neuron_index": 10, "strength_score": 0.9},
        {"neuron_index": 11, "strength_score": 0.7},
        {"neuron_index": 12, "strength_score": 0.3},
        {"neuron_index": 13, "strength_score": 0.1},
        {"neuron_index": 14, "strength_score": 0.05},
    ]
    strong, weak = split_strong_weak(rows)
    assert strong == [10, 11, 12]
    assert weak == [13, 14]


def test_build_probe_subsets_contains_mixed_variants() -> None:
    rows = build_probe_subsets([100, 101, 102], [200, 201], [100, 101, 102, 200, 201])
    labels = {row["label"] for row in rows}
    indices = {tuple(row["indices"]) for row in rows}
    assert "strong_top1" in labels
    assert (100, 101, 102) in indices
    assert (200, 201) in indices
    assert "union_full" in labels
    assert "mix_top1_plus_200" in labels
    assert "mix_top2_plus_201" in labels
