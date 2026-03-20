from __future__ import annotations

from stage56_concept_cross_asset_closure import build_concept_cross_asset_closure_summary


def test_concept_cross_asset_closure_is_positive() -> None:
    summary = build_concept_cross_asset_closure_summary()
    hm = summary["headline_metrics"]

    assert hm["support_consensus"] > 0.5
    assert hm["closure_support"] > 0.5
    assert hm["closure_margin"] > 0.0
