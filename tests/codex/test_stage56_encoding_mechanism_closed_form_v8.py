from __future__ import annotations

from stage56_encoding_mechanism_closed_form_v8 import build_encoding_mechanism_closed_form_v8_summary


def test_encoding_mechanism_closed_form_v8_positive() -> None:
    summary = build_encoding_mechanism_closed_form_v8_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_margin_v8"] > 0.0
    assert hm["structure_term_v8"] > hm["pressure_term_v8"]
