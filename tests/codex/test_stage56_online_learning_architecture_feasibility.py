from __future__ import annotations

from tests.codex.stage56_online_learning_architecture_feasibility import (
    build_online_learning_architecture_feasibility_summary,
)


def test_online_learning_architecture_feasibility_is_positive() -> None:
    hm = build_online_learning_architecture_feasibility_summary()["headline_metrics"]
    assert hm["language_capability_readiness"] > 0.0
    assert hm["online_stability_readiness"] > 0.0
    assert hm["architecture_feasibility"] > 0.0
    assert hm["production_gap"] >= 0.0
