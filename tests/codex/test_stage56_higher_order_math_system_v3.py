from __future__ import annotations

from stage56_higher_order_math_system_v3 import build_summary


def test_build_summary_exposes_morphism_hint() -> None:
    summary = build_summary({"equations": {"general_equation": "x"}, "state_dictionary": {"G_final": "kernel_v4"}})
    assert "morphism_hint" in summary["system_objects"]
    assert summary["state_dictionary"]["G_final"] == "kernel_v4"
