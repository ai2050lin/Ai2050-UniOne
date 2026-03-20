from __future__ import annotations

from stage56_concept_formation_closed_form_v3 import build_concept_formation_closed_form_v3_summary


def test_concept_formation_closed_form_v3_is_positive() -> None:
    summary = build_concept_formation_closed_form_v3_summary()
    hm = summary["headline_metrics"]

    assert hm["anchor_chart_term_v3"] > 0.0
    assert hm["strengthened_fiber_term_v3"] > 0.0
    assert hm["cross_asset_term_v3"] > 0.0
    assert hm["concept_margin_v3"] > hm["pressure_term_v3"]
