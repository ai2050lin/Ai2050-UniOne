from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v41 import build_encoding_mechanism_closed_form_v41_summary


def test_encoding_mechanism_closed_form_v41_improves_v40() -> None:
    hm = build_encoding_mechanism_closed_form_v41_summary()["headline_metrics"]
    assert hm["feature_term_v41"] > 2450.7135655990833
    assert hm["structure_term_v41"] > 7522.140878796387
    assert hm["learning_term_v41"] > 58661.76323991915
    assert hm["encoding_margin_v41"] > 68626.7953299929
