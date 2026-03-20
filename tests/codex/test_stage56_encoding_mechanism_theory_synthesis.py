from __future__ import annotations

try:
    from tests.codex.stage56_encoding_mechanism_theory_synthesis import build_encoding_mechanism_theory_synthesis_summary
except ModuleNotFoundError:
    from stage56_encoding_mechanism_theory_synthesis import build_encoding_mechanism_theory_synthesis_summary


def test_theory_synthesis_margin_positive() -> None:
    summary = build_encoding_mechanism_theory_synthesis_summary()
    hm = summary["headline_metrics"]
    assert hm["mechanism_strength"] > hm["pressure_strength"]
    assert hm["theory_margin"] > 0.0
