from __future__ import annotations

from tests.codex.stage56_dnn_brain_math_theory_synthesis import build_dnn_brain_math_theory_synthesis_summary


def test_dnn_brain_math_theory_synthesis_is_positive() -> None:
    hm = build_dnn_brain_math_theory_synthesis_summary()["headline_metrics"]
    assert hm["dnn_language_core"] > 0.0
    assert hm["brain_encoding_core"] > 0.0
    assert hm["math_system_core"] > 0.0
    assert hm["theory_bridge_strength"] > 0.0
