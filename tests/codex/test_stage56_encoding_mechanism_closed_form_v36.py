from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v36 import build_encoding_mechanism_closed_form_v36_summary


def test_encoding_mechanism_closed_form_v36_improves_v35() -> None:
    hm = build_encoding_mechanism_closed_form_v36_summary()["headline_metrics"]
    assert hm["feature_term_v36"] > 855.9152120001014
    assert hm["structure_term_v36"] > 2697.2637802949853
    assert hm["learning_term_v36"] > 8423.507241443283
    assert hm["pressure_term_v36"] < 8.32207639242729
    assert hm["encoding_margin_v36"] > 11968.364157345943
