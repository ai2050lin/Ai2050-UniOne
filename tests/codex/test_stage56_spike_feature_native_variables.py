from __future__ import annotations

from stage56_spike_feature_native_variables import build_spike_feature_native_variables_summary


def test_spike_feature_native_variables_positive() -> None:
    summary = build_spike_feature_native_variables_summary()
    hm = summary["headline_metrics"]
    assert hm["native_seed"] > hm["native_inhibition"]
    assert hm["native_extraction_margin"] > 0.0
    assert hm["native_selectivity"] > 0.0
