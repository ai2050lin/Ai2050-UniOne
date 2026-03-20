from __future__ import annotations

from stage56_concept_local_chart_expansion import build_concept_local_chart_expansion_summary


def test_concept_local_chart_expansion_summary_is_positive() -> None:
    summary = build_concept_local_chart_expansion_summary()
    hm = summary["headline_metrics"]

    assert hm["family_count"] == 3
    assert hm["mean_anchor_strength"] > 0.5
    assert hm["mean_chart_support"] > 0.0
    assert hm["mean_separation_gap"] > 0.0


def test_each_family_chart_has_low_reconstruction_error() -> None:
    summary = build_concept_local_chart_expansion_summary()
    for row in summary["family_charts"].values():
        assert row["reconstruction_error_mean"] < 0.2
