from __future__ import annotations

from tests.codex.stage56_cross_modal_unification_strengthening import (
    build_cross_modal_unification_strengthening_summary,
)


def test_cross_modal_unification_strengthening_improves_bridge() -> None:
    hm = build_cross_modal_unification_strengthening_summary()["headline_metrics"]
    assert hm["cross_modal_unification_stable"] > 0.6957591237089628
    assert hm["modality_residual_stable"] < 0.30424087629103724
