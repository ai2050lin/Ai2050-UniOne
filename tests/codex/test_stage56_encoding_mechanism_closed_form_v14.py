from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_encoding_mechanism_closed_form_v14 import build_encoding_mechanism_closed_form_v14_summary


def test_encoding_mechanism_closed_form_v14_margin_is_large_and_ordered() -> None:
    summary = build_encoding_mechanism_closed_form_v14_summary()
    hm = summary["headline_metrics"]
    assert hm["feature_term_v14"] > hm["pressure_term_v14"]
    assert hm["structure_term_v14"] > hm["feature_term_v14"]
    assert hm["learning_term_v14"] > hm["structure_term_v14"]
    assert hm["encoding_margin_v14"] > hm["learning_term_v14"]
