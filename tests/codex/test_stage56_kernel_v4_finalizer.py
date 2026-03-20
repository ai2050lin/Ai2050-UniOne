from __future__ import annotations

from stage56_kernel_v4_finalizer import build_summary


def test_build_summary_prefers_kernel_v4_when_sample_and_corpus_are_positive() -> None:
    summary = build_summary(
        {"signs": {"a": "positive", "b": "positive", "c": "positive"}},
        {"sign_matrix": {"G_corpus_proxy": {"x": "positive", "y": "positive"}}},
    )
    assert summary["final_decision"] == "G_final = kernel_v4"
    assert summary["final_score"] == 1.0
