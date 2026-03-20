from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_feature_dominance_absolute_lock import build_feature_dominance_absolute_lock_summary


def test_feature_dominance_absolute_lock_is_positive() -> None:
    summary = build_feature_dominance_absolute_lock_summary()
    hm = summary["headline_metrics"]
    assert hm["absolute_margin"] > 0.0
    assert hm["absolute_ratio"] > 1.0
    assert hm["absolute_gain"] > hm["absolute_gap"]
