from __future__ import annotations

from stage56_feature_primary_dominance import build_feature_primary_dominance_summary


def test_feature_primary_dominance_positive() -> None:
    summary = build_feature_primary_dominance_summary()
    hm = summary["headline_metrics"]
    assert hm["dominance_margin"] > 0.0
    assert hm["dominance_ratio"] > 1.0
