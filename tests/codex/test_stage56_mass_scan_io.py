from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_mass_scan_io import row_term, scan_term_rows  # noqa: E402


def test_scan_term_rows_prefers_term_records():
    payload = {
        "term_records": [{"term": "run", "category": "action"}],
        "noun_records": [{"noun": "apple", "category": "fruit"}],
    }
    rows = scan_term_rows(payload)
    assert rows == [{"term": "run", "category": "action"}]
    assert row_term(rows[0]) == "run"


def test_scan_term_rows_falls_back_to_noun_records():
    payload = {"noun_records": [{"noun": "apple", "category": "fruit"}]}
    rows = scan_term_rows(payload)
    assert rows == [{"noun": "apple", "category": "fruit"}]
    assert row_term(rows[0]) == "apple"
