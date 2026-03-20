from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_encoding_mechanism_closed_form_v11 import build_encoding_mechanism_closed_form_v11_summary


def test_encoding_mechanism_closed_form_v11_margin_is_large_and_positive() -> None:
    summary = build_encoding_mechanism_closed_form_v11_summary()
    hm = summary["headline_metrics"]
    assert hm["feature_term_v11"] > hm["pressure_term_v11"]
    assert hm["structure_term_v11"] > hm["feature_term_v11"]
    assert hm["learning_term_v11"] > hm["structure_term_v11"]
    assert hm["encoding_margin_v11"] > hm["learning_term_v11"]
