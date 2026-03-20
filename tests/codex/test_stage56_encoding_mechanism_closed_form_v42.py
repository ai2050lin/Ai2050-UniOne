from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v42 import build_encoding_mechanism_closed_form_v42_summary


def test_encoding_mechanism_closed_form_v42_improves_v41() -> None:
    hm = build_encoding_mechanism_closed_form_v42_summary()["headline_metrics"]
    assert hm["feature_term_v42"] > 2503.732977969676
    assert hm["structure_term_v42"] > 7828.4313938212445
    assert hm["learning_term_v42"] > 96180.13967863844
    assert hm["encoding_margin_v42"] > 106503.83524707537
