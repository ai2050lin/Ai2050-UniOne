from __future__ import annotations

from stage56_feature_primary_threshold_closure import build_feature_primary_threshold_closure_summary


def test_feature_primary_threshold_closure_positive() -> None:
    summary = build_feature_primary_threshold_closure_summary()
    hm = summary["headline_metrics"]
    assert hm["primary_threshold_margin"] > 0.0
    assert hm["primary_threshold_ratio"] > 1.0
