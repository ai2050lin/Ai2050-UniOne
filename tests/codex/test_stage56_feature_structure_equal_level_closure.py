from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_feature_structure_equal_level_closure import build_feature_structure_equal_level_closure_summary


def test_feature_structure_equal_level_closure_is_balanced() -> None:
    summary = build_feature_structure_equal_level_closure_summary()
    hm = summary["headline_metrics"]

    assert hm["equal_geometric_core"] > 0.0
    assert hm["equalized_ratio_v3"] == 1.0
    assert hm["equalized_gap_v3"] == 0.0
    assert hm["equalization_confidence"] > 0.9
