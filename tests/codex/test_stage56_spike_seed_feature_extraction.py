from __future__ import annotations

from stage56_spike_seed_feature_extraction import build_spike_seed_feature_extraction_summary


def test_spike_seed_feature_extraction_margin_positive() -> None:
    summary = build_spike_seed_feature_extraction_summary()
    hm = summary["headline_metrics"]
    assert hm["spike_seed_drive"] > hm["inhibitory_filter"]
    assert hm["feature_extraction_margin"] > 0.0
