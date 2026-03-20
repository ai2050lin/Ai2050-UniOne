from __future__ import annotations

from stage56_local_differential_fiber_strengthening import build_local_differential_fiber_strengthening_summary


def test_local_differential_fiber_strengthening_is_positive() -> None:
    summary = build_local_differential_fiber_strengthening_summary()
    hm = summary["headline_metrics"]

    assert hm["family_count"] == 3
    assert hm["mean_strengthened_local_fiber"] > 0.0
    assert hm["apple_strengthened_local_margin"] > 0.05
