from __future__ import annotations

from stage56_learning_dynamics_bridge import build_summary


def test_build_summary_creates_learning_equations() -> None:
    summary = build_summary(
        {
            "native_proxy_summary": {
                "G_native_proxy": {"correlations": {"a": 0.2, "b": 0.1}},
                "L_base_native_proxy": {"correlations": {"a": -0.3, "b": -0.2}},
                "L_select_native_proxy": {"correlations": {"a": -0.1, "b": -0.2}},
            }
        },
        {"support": {"strict_closure_confidence": 0.5}},
        {"system_objects": {"state_bundle": "Z"}},
    )
    assert "atlas_update" in summary["learning_equations"]
    assert len(summary["emergence_order"]) == 3
