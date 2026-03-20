from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v35 import build_encoding_mechanism_closed_form_v35_summary


def test_encoding_mechanism_closed_form_v35_improves_v34() -> None:
    summary = build_encoding_mechanism_closed_form_v35_summary()
    hm = summary["headline_metrics"]

    assert hm["feature_term_v35"] > 672.7188821337269
    assert hm["structure_term_v35"] > 2195.944763504757
    assert hm["learning_term_v35"] > 7705.836653507162
    assert hm["pressure_term_v35"] < 8.957444994848915
    assert hm["encoding_margin_v35"] > 10565.542854150797
