from __future__ import annotations

from tests.codex.stage56_icspb_unification_high_closure import build_icspb_unification_high_closure_summary


def test_icspb_unification_high_closure_improves_reinforced_version() -> None:
    hm = build_icspb_unification_high_closure_summary()["headline_metrics"]
    assert hm["unification_high_stability"] > 0.684075606953117
    assert hm["support_gap_high"] < 0.3229967585702239
    assert hm["high_closure_gain"] > 0.0
