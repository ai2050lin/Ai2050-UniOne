from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_encoding_mechanism_closed_form_v13 import build_encoding_mechanism_closed_form_v13_summary


def test_encoding_mechanism_closed_form_v13_margin_is_large_and_ordered() -> None:
    summary = build_encoding_mechanism_closed_form_v13_summary()
    hm = summary["headline_metrics"]
    assert hm["feature_term_v13"] > hm["pressure_term_v13"]
    assert hm["structure_term_v13"] > hm["feature_term_v13"]
    assert hm["learning_term_v13"] > hm["structure_term_v13"]
    assert hm["encoding_margin_v13"] > hm["learning_term_v13"]
