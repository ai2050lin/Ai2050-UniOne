from __future__ import annotations

from stage56_prototype_online_learning_experiment import build_summary


def test_build_summary_emits_forgetting_and_novel_gain() -> None:
    summary, _ = build_summary(order=7, base_cut=4, epochs=2, inject_steps=1, batch_size=8, seed=1)
    assert "forgetting" in summary["deltas"]
    assert "novel_accuracy_delta" in summary["deltas"]
