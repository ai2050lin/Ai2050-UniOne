from __future__ import annotations

from stage56_encoding_mechanism_closed_form_v9 import build_encoding_mechanism_closed_form_v9_summary


def test_encoding_mechanism_closed_form_v9_positive() -> None:
    summary = build_encoding_mechanism_closed_form_v9_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_margin_v9"] > 0.0
    assert hm["structure_term_v9"] > hm["pressure_term_v9"]
