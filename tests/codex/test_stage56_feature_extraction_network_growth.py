from __future__ import annotations

from stage56_feature_extraction_network_growth import build_feature_extraction_network_growth_summary


def test_feature_extraction_network_growth_positive() -> None:
    summary = build_feature_extraction_network_growth_summary()
    hm = summary["headline_metrics"]
    assert hm["structure_embedding_drive"] > hm["structure_pressure"]
    assert hm["network_structure_margin"] > 0.0
    assert hm["global_steady_drive"] > 0.0
