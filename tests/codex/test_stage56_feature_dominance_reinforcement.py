from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_feature_dominance_reinforcement import build_feature_dominance_reinforcement_summary


def test_feature_dominance_reinforcement_is_positive() -> None:
    summary = build_feature_dominance_reinforcement_summary()
    hm = summary["headline_metrics"]
    assert hm["reinforced_gain"] > hm["reinforced_gap"]
    assert hm["reinforced_margin"] > 0.0
    assert hm["reinforced_ratio"] > 1.0
