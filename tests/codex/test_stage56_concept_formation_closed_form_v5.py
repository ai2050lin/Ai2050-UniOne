from __future__ import annotations

from stage56_concept_formation_closed_form_v5 import build_concept_formation_closed_form_v5_summary


def test_concept_formation_closed_form_v5_margin_positive() -> None:
    summary = build_concept_formation_closed_form_v5_summary()
    hm = summary["headline_metrics"]
    assert hm["concept_margin_v5"] > hm["anchor_chart_term_v5"]
    assert hm["concept_margin_v5"] > 0.0
    assert hm["pressure_term_v5"] < hm["anchor_chart_term_v5"]
