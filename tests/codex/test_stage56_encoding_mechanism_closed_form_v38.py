from __future__ import annotations

from tests.codex.stage56_encoding_mechanism_closed_form_v38 import build_encoding_mechanism_closed_form_v38_summary


def test_encoding_mechanism_closed_form_v38_improves_v37() -> None:
    hm = build_encoding_mechanism_closed_form_v38_summary()["headline_metrics"]
    assert hm["feature_term_v38"] > 1476.3470392412846
    assert hm["structure_term_v38"] > 4847.289984130736
    assert hm["learning_term_v38"] > 14869.849823069138
    assert hm["pressure_term_v38"] < 7.991422172104651
    assert hm["encoding_margin_v38"] > 21185.495424269055
