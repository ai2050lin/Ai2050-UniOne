from __future__ import annotations

from stage56_hierarchy_emergence_analysis import build_summary


def test_build_summary_has_three_phases() -> None:
    summary = build_summary(
        {"learning_state": {"L_select_instability": 0.2, "L_base_load": 0.3}},
        {"final_score": 1.0},
        {"closure_confidence": 0.5},
    )
    assert len(summary["phase_order"]) == 3
