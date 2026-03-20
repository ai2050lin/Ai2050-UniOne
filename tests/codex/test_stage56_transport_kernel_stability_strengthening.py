from __future__ import annotations

from tests.codex.stage56_transport_kernel_stability_strengthening import (
    build_transport_kernel_stability_strengthening_summary,
)


def test_transport_kernel_stability_strengthening_improves_reinforced_version() -> None:
    hm = build_transport_kernel_stability_strengthening_summary()["headline_metrics"]
    assert hm["transport_kernel_stability_stable"] > 0.3592930467851382
    assert hm["weakest_channel_stable"] > 0.2833261737434117
    assert hm["stability_lift"] > 0.0
