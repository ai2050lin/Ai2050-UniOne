from __future__ import annotations

from stage56_concept_formation_closed_form_v2 import build_concept_formation_closed_form_v2_summary


def test_concept_formation_closed_form_v2_is_positive() -> None:
    summary = build_concept_formation_closed_form_v2_summary()
    hm = summary["headline_metrics"]

    assert hm["family_anchor_term"] > 0.5
    assert hm["local_chart_term"] > 0.0
    assert hm["local_fiber_term"] > 0.0
    assert hm["concept_margin_v2"] > hm["formation_pressure_term"]
