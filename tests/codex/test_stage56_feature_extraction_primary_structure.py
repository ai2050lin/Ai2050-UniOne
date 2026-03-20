from __future__ import annotations

from stage56_feature_extraction_primary_structure import build_feature_extraction_primary_structure_summary


def test_feature_extraction_primary_structure_positive() -> None:
    summary = build_feature_extraction_primary_structure_summary()
    hm = summary["headline_metrics"]
    assert hm["primary_feature_core"] > 0.0
    assert hm["feature_structure_support"] > 0.0
    assert hm["feature_primary_ratio"] > 1.0
