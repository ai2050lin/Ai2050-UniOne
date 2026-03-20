from __future__ import annotations

from tests.codex.stage56_icspb_unification_reinforcement import build_icspb_unification_reinforcement_summary


def test_icspb_unification_reinforcement_improves_closure() -> None:
    summary = build_icspb_unification_reinforcement_summary()
    hm = summary["headline_metrics"]

    assert hm["unification_stability_reinforced"] > 0.6196997759360026
    assert hm["support_gap_reinforced"] < 0.4081430721427013
    assert hm["unification_gain"] > 0.0
