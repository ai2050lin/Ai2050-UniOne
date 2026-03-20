from __future__ import annotations

from tests.codex.stage56_prototype_network_readiness import build_prototype_network_readiness_summary


def test_prototype_network_readiness_is_bounded() -> None:
    hm = build_prototype_network_readiness_summary()["headline_metrics"]
    assert 0.0 <= hm["language_stack_readiness"] <= 1.0
    assert 0.0 <= hm["online_learning_readiness"] <= 1.0
    assert 0.0 <= hm["prototype_network_readiness"] <= 1.0
    assert hm["agi_delivery_gap"] >= 0.0
