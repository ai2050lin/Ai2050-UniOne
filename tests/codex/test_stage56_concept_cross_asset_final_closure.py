from __future__ import annotations

from stage56_concept_cross_asset_final_closure import build_concept_cross_asset_final_closure_summary


def test_concept_cross_asset_final_closure_margin_positive() -> None:
    summary = build_concept_cross_asset_final_closure_summary()
    hm = summary["headline_metrics"]
    assert hm["final_closure_support"] > hm["final_gap_penalty"]
    assert hm["final_closure_margin"] > 0.0
    assert 0.0 < hm["closure_to_margin_ratio"] < 1.0
