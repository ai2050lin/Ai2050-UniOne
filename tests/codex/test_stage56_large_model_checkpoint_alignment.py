try:
    from tests.codex.stage56_large_model_checkpoint_alignment import build_alignment_summary
except ModuleNotFoundError:
    from stage56_large_model_checkpoint_alignment import build_alignment_summary


def test_build_alignment_summary_has_cases() -> None:
    summary = build_alignment_summary()
    assert "headline_metrics" in summary
    assert summary["case_count"] >= 4
    assert summary["headline_metrics"]["boundary_mean_step"] >= summary["headline_metrics"]["atlas_mean_step"]
