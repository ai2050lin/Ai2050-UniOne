from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v40 import build_encoding_mechanism_closed_form_v40_summary


def test_encoding_mechanism_closed_form_v40_improves_v39() -> None:
    hm = build_encoding_mechanism_closed_form_v40_summary()["headline_metrics"]
    assert hm["feature_term_v40"] > 2269.1792274065588
    assert hm["structure_term_v40"] > 7238.106153452666
    assert hm["learning_term_v40"] > 35402.18770065925
    assert hm["encoding_margin_v40"] > 44902.18576476588
