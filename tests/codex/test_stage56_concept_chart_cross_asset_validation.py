from __future__ import annotations

from stage56_concept_chart_cross_asset_validation import build_concept_chart_cross_asset_validation_summary


def test_concept_chart_cross_asset_support_is_positive() -> None:
    summary = build_concept_chart_cross_asset_validation_summary()
    hm = summary["headline_metrics"]

    assert hm["chart_family_support"] > 0.5
    assert hm["chart_separation_support"] > 0.5
    assert hm["cross_asset_support_v2"] > 0.5
    assert hm["support_gap_v2"] < 1.5
