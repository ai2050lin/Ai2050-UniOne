from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_encoding_mechanism_closed_form_v34 import build_encoding_mechanism_closed_form_v34_summary


def test_encoding_mechanism_closed_form_v34_positive() -> None:
    summary = build_encoding_mechanism_closed_form_v34_summary()
    hm = summary["headline_metrics"]

    assert hm["feature_term_v34"] > 0.0
    assert hm["structure_term_v34"] > hm["feature_term_v34"]
    assert hm["learning_term_v34"] > hm["structure_term_v34"]
    assert hm["pressure_term_v34"] > 0.0
    assert hm["encoding_margin_v34"] > hm["learning_term_v34"]
