from __future__ import annotations

from stage56_spiking_dynamics_bridge_v3 import build_summary


def test_build_summary_has_spike_state() -> None:
    summary = build_summary(
        {"before_injection": {}, "after_injection": {"general_norm_mean": 1.0, "disc_mean": 0.4}, "deltas": {"base_accuracy_delta": -0.1, "strict_gate_shift": 0.2}},
        {"delta": {"boundary_grad_delta": 0.3}},
    )
    assert "excitatory_drive" in summary["spike_bridge_state"]
