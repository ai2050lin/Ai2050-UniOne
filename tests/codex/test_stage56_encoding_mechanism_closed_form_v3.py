from __future__ import annotations

try:
    from tests.codex.stage56_encoding_mechanism_closed_form_v3 import (
        build_encoding_mechanism_closed_form_v3_summary,
    )
except ModuleNotFoundError:
    from stage56_encoding_mechanism_closed_form_v3 import build_encoding_mechanism_closed_form_v3_summary


def test_closed_form_margin_v3_exceeds_zero() -> None:
    summary = build_encoding_mechanism_closed_form_v3_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_kernel_v3"] > 0.0
    assert hm["structure_growth_v3"] > 0.0
    assert hm["closed_form_margin_v3"] > 0.0
