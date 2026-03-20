from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v44 import build_encoding_mechanism_closed_form_v44_summary


def test_encoding_mechanism_closed_form_v44_improves_v43() -> None:
    hm = build_encoding_mechanism_closed_form_v44_summary()["headline_metrics"]
    assert hm["feature_term_v44"] > 2632.2708336900228
    assert hm["structure_term_v44"] > 8227.01640271567
    assert hm["learning_term_v44"] > 220512.4498230946
    assert hm["encoding_margin_v44"] > 231361.8897883955
