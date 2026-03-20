from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_feature_dominance_irreversibility import build_feature_dominance_irreversibility_summary


def test_feature_dominance_irreversibility_is_positive() -> None:
    summary = build_feature_dominance_irreversibility_summary()
    hm = summary["headline_metrics"]
    assert hm["irreversible_margin"] > 0.0
    assert hm["irreversible_ratio"] > 1.0
    assert hm["irreversible_gain"] > hm["irreversible_gap"]
