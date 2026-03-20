from __future__ import annotations

from stage56_concept_formation_closed_form_v4 import build_concept_formation_closed_form_v4_summary


def test_concept_formation_closed_form_v4_is_positive() -> None:
    summary = build_concept_formation_closed_form_v4_summary()
    hm = summary["headline_metrics"]

    assert hm["anchor_chart_term_v4"] > 0.0
    assert hm["local_primary_term_v4"] > 0.0
    assert hm["circuit_term_v4"] > 0.0
    assert hm["concept_margin_v4"] > hm["pressure_term_v4"]
