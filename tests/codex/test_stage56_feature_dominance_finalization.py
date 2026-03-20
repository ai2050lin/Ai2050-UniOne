from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_feature_dominance_finalization import build_feature_dominance_finalization_summary


def test_feature_dominance_finalization_strengthens_margin() -> None:
    summary = build_feature_dominance_finalization_summary()
    hm = summary["headline_metrics"]
    assert hm["final_margin"] > 0.0
    assert hm["final_ratio"] > 1.0
    assert hm["final_gain"] > hm["final_gap"]
