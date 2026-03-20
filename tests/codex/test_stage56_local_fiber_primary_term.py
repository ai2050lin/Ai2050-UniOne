from __future__ import annotations

from stage56_local_fiber_primary_term import build_local_fiber_primary_term_summary


def test_local_fiber_primary_term_is_positive() -> None:
    summary = build_local_fiber_primary_term_summary()
    hm = summary["headline_metrics"]

    assert hm["fiber_gain"] > 0.0
    assert hm["apple_primary_local_term"] > 0.05
    assert hm["local_primary_margin"] > hm["fiber_gain"]
