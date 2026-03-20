from __future__ import annotations

from tests.codex.stage56_falsifiability_closure import build_falsifiability_closure_summary


def test_falsifiability_closure_is_bounded() -> None:
    hm = build_falsifiability_closure_summary()["headline_metrics"]
    assert hm["testability_strength"] > 0.0
    assert hm["equation_compactness"] > 0.0
    assert 0.0 <= hm["falsifiability_closure"] <= 1.0
    assert hm["residual_nonfalsifiable"] >= 0.0
