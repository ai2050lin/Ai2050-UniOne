from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_encoding_mechanism_closed_form_v32 import build_encoding_mechanism_closed_form_v32_summary


def test_encoding_mechanism_closed_form_v32_positive() -> None:
    summary = build_encoding_mechanism_closed_form_v32_summary()
    hm = summary["headline_metrics"]

    assert hm["feature_term_v32"] > 0.0
    assert hm["structure_term_v32"] > hm["feature_term_v32"]
    assert hm["learning_term_v32"] > hm["structure_term_v32"]
    assert hm["encoding_margin_v32"] > hm["learning_term_v32"]
