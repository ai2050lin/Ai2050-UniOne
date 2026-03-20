from __future__ import annotations

from stage56_training_trajectory_bridge import build_summary


def test_build_summary_extracts_three_phase_groups() -> None:
    summary = build_summary(
        {
            "history": [
                {"mean_train_loss": 10.0, "eval_loss": 12.0, "semantic_benchmark_score": 0.1, "generation_quality_score": 0.2},
                {"mean_train_loss": 8.0, "eval_loss": 10.0, "semantic_benchmark_score": 0.2, "generation_quality_score": 0.3},
                {"mean_train_loss": 6.0, "eval_loss": 8.0, "semantic_benchmark_score": 0.3, "generation_quality_score": 0.4},
            ]
        },
        {
            "Transformer": [{"epoch": 1, "loss": 5.0, "accuracy": 10.0}, {"epoch": 2, "loss": 4.0, "accuracy": 20.0}],
            "FiberNet": [{"epoch": 1, "loss": 6.0, "accuracy": 5.0}, {"epoch": 2, "loss": 3.0, "accuracy": 25.0}],
        },
    )
    assert "icspb_phase" in summary
    assert "transformer_phase" in summary
    assert "fibernet_phase" in summary
