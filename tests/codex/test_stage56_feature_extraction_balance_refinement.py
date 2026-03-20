from __future__ import annotations

from stage56_feature_extraction_balance_refinement import build_feature_extraction_balance_refinement_summary


def test_feature_extraction_balance_refinement_positive() -> None:
    summary = build_feature_extraction_balance_refinement_summary()
    hm = summary["headline_metrics"]
    assert hm["balanced_feature_gain"] > hm["seed_normalized"]
    assert hm["feature_balance_margin"] > 0.0
    assert hm["extraction_balance_ratio"] > 1.0
