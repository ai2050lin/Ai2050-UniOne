from __future__ import annotations

from stage56_weight_update_geometry import build_summary


def test_build_summary_emits_three_geometry_updates() -> None:
    summary = build_summary(
        {"learning_state": {"atlas_learning_drive": 0.1, "frontier_learning_drive": 0.2, "closure_learning_drive": 0.3}},
        {"equations": {"general_equation": "x"}},
    )
    assert set(summary["geometry_updates"].keys()) == {"atlas_shift", "frontier_shift", "boundary_shift"}
