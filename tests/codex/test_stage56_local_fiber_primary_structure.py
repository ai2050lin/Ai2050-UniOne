from __future__ import annotations

from stage56_local_fiber_primary_structure import build_local_fiber_primary_structure_summary


def test_local_fiber_primary_structure_strengthens_margin() -> None:
    summary = build_local_fiber_primary_structure_summary()
    hm = summary["headline_metrics"]
    assert hm["local_primary_structure"] > hm["fiber_structure_gain"]
    assert hm["local_primary_structure"] > 0.0
