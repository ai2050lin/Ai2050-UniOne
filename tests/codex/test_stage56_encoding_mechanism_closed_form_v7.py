from __future__ import annotations

from stage56_encoding_mechanism_closed_form_v7 import build_encoding_mechanism_closed_form_v7_summary


def test_encoding_mechanism_closed_form_v7_positive() -> None:
    summary = build_encoding_mechanism_closed_form_v7_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_margin_v7"] > 0.0
    assert hm["structure_term_v7"] > hm["pressure_term_v7"]
