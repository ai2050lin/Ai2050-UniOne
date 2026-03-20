from __future__ import annotations

from stage56_checkpoint_geometry_bridge import build_summary


def test_build_summary_aligns_three_geometry_axes() -> None:
    summary = build_summary(
        {"icspb_phase": {"base_phase": 1.0, "general_phase": 2.0, "strict_phase": 3.0}},
        {"learning_state": {"atlas_learning_drive": 0.1, "frontier_learning_drive": 0.2, "closure_learning_drive": 0.3}},
    )
    assert summary["trajectory_alignment"]["atlas_alignment"] == 1.1
