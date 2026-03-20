from __future__ import annotations

from stage56_attribute_fiber_nativeization import build_attribute_fiber_nativeization_summary


def test_attribute_fiber_nativeization_splits_anchor_and_local_layers() -> None:
    summary = build_attribute_fiber_nativeization_summary()
    hm = summary["headline_metrics"]

    assert hm["mean_anchor_bundle_strength"] > 0.0
    assert hm["mean_local_bundle_strength"] > 0.0
    assert hm["apple_anchor_attribute_count"] >= 1
    assert hm["apple_local_attribute_count"] >= 1


def test_apple_round_and_elongated_are_opposite() -> None:
    summary = build_attribute_fiber_nativeization_summary()
    hm = summary["headline_metrics"]

    assert hm["apple_round_local_coeff"] > 0.0
    assert hm["apple_elongated_local_coeff"] < 0.0
