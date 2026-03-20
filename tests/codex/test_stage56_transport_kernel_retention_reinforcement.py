from __future__ import annotations

from tests.codex.stage56_transport_kernel_retention_reinforcement import (
    build_transport_kernel_retention_reinforcement_summary,
)


def test_transport_kernel_retention_reinforcement_improves_stability() -> None:
    summary = build_transport_kernel_retention_reinforcement_summary()
    hm = summary["headline_metrics"]

    assert hm["transport_kernel_stability_reinforced"] > 0.13769209710067937
    assert hm["update_retention_reinforced"] > 0.05303346099257981
    assert hm["retention_recovery_margin"] > 0.0
    assert hm["weakest_channel_floor"] > 0.05
