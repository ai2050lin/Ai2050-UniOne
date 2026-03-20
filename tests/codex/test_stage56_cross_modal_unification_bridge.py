from __future__ import annotations

from tests.codex.stage56_cross_modal_unification_bridge import build_cross_modal_unification_bridge_summary


def test_cross_modal_unification_bridge_is_bounded() -> None:
    hm = build_cross_modal_unification_bridge_summary()["headline_metrics"]
    assert hm["language_to_general_transfer"] > 0.0
    assert hm["modality_extension_strength"] > 0.0
    assert 0.0 <= hm["cross_modal_unification_strength"] <= 1.0
    assert hm["modality_residual"] >= 0.0
