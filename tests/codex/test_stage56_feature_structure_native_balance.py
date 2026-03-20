from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_feature_structure_native_balance import build_feature_structure_native_balance_summary


def test_feature_structure_native_balance_positive() -> None:
    summary = build_feature_structure_native_balance_summary()
    hm = summary["headline_metrics"]

    assert hm["bridge_gain"] > 1.0
    assert hm["native_balanced_feature_v2"] > 0.0
    assert hm["native_balanced_structure_v2"] > 0.0
    assert hm["native_balance_ratio_v2"] > 0.5
    assert hm["native_balance_gap_v2"] >= 0.0
