from __future__ import annotations

from tests.codex.stage56_training_terminal_rule import build_training_terminal_rule_summary


def test_training_terminal_rule_is_bounded() -> None:
    hm = build_training_terminal_rule_summary()["headline_metrics"]
    assert 0.0 <= hm["terminal_update_strength"] <= 1.0
    assert 0.0 <= hm["terminal_stability_guard"] <= 1.0
    assert 0.0 <= hm["prototype_trainability"] <= 1.0
    assert 0.0 <= hm["training_terminal_readiness"] <= 1.0
    assert hm["terminal_training_gap"] >= 0.0
