from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_encoding_mechanism_closed_form_v12 import build_encoding_mechanism_closed_form_v12_summary


def test_encoding_mechanism_closed_form_v12_margin_is_large_and_ordered() -> None:
    summary = build_encoding_mechanism_closed_form_v12_summary()
    hm = summary["headline_metrics"]
    assert hm["feature_term_v12"] > hm["pressure_term_v12"]
    assert hm["structure_term_v12"] > hm["feature_term_v12"]
    assert hm["learning_term_v12"] > hm["structure_term_v12"]
    assert hm["encoding_margin_v12"] > hm["learning_term_v12"]
