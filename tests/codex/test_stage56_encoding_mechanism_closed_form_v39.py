from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v39 import build_encoding_mechanism_closed_form_v39_summary


def test_encoding_mechanism_closed_form_v39_improves_v38() -> None:
    hm = build_encoding_mechanism_closed_form_v39_summary()["headline_metrics"]
    assert hm["feature_term_v39"] > 2007.6913589405249
    assert hm["structure_term_v39"] > 6693.04430957211
    assert hm["learning_term_v39"] > 22896.43379335998
    assert hm["pressure_term_v39"] < 7.489310246705163
    assert hm["encoding_margin_v39"] > 31589.68015162591
