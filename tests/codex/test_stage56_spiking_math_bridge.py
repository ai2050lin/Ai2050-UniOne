from __future__ import annotations

from stage56_spiking_math_bridge import build_spiking_bridge


def test_build_spiking_bridge_exposes_time_window_and_attractor_view() -> None:
    out = build_spiking_bridge()
    assert out["record_type"] == "stage56_spiking_math_bridge_summary"
    assert "时间窗" in out["main_judgment"]
    assert "吸引域" in out["proto_spiking_equation"] or "Attractor_static" in out["proto_spiking_equation"]
    assert len(out["mappings"]) == 6
