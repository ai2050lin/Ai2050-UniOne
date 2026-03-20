from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v37 import build_encoding_mechanism_closed_form_v37_summary


def test_encoding_mechanism_closed_form_v37_improves_v36() -> None:
    hm = build_encoding_mechanism_closed_form_v37_summary()["headline_metrics"]
    assert hm["feature_term_v37"] > 1160.3562081330513
    assert hm["structure_term_v37"] > 3615.85670440388
    assert hm["learning_term_v37"] > 9297.617002672361
    assert hm["pressure_term_v37"] < 8.123435601379345
    assert hm["encoding_margin_v37"] > 14065.706479607912
