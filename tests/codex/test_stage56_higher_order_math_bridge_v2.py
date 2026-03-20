from __future__ import annotations

from stage56_higher_order_math_bridge_v2 import build_summary


def test_build_summary_exposes_bridge_objects() -> None:
    summary = build_summary({"closed_equations": {"general_equation": "x"}})
    bridge = dict(summary["bridge_objects"])
    assert "Z_general" in str(bridge["state_bundle"])
    assert "C_drive" in str(bridge["channel_bundle"])
