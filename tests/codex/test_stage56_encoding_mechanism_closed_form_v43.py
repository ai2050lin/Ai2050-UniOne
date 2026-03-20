from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v43 import build_encoding_mechanism_closed_form_v43_summary


def test_encoding_mechanism_closed_form_v43_improves_v42() -> None:
    hm = build_encoding_mechanism_closed_form_v43_summary()["headline_metrics"]
    assert hm["feature_term_v43"] > 2573.4127804798127
    assert hm["structure_term_v43"] > 8046.430458207543
    assert hm["learning_term_v43"] > 141657.9354177618
    assert hm["encoding_margin_v43"] > 152268.54457513907
