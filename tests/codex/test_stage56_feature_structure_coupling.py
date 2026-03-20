from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_feature_structure_coupling import build_feature_structure_coupling_summary


def test_feature_structure_coupling_is_positive() -> None:
    summary = build_feature_structure_coupling_summary()
    hm = summary["headline_metrics"]
    assert hm["feature_to_structure"] > hm["feature_to_circuit"]
    assert hm["structure_feedback"] > 1.0
    assert hm["coupling_margin"] > 0.0
