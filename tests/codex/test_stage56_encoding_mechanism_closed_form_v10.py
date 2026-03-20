from __future__ import annotations

from stage56_encoding_mechanism_closed_form_v10 import build_encoding_mechanism_closed_form_v10_summary


def test_encoding_mechanism_closed_form_v10_positive() -> None:
    summary = build_encoding_mechanism_closed_form_v10_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_margin_v10"] > 0.0
    assert hm["structure_term_v10"] > hm["pressure_term_v10"]
