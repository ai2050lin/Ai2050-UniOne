from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_feature_structure_balance_normalization import (
    build_feature_structure_balance_normalization_summary,
)


def test_feature_structure_balance_normalization_reduces_ratio_gap() -> None:
    summary = build_feature_structure_balance_normalization_summary()
    hm = summary["headline_metrics"]

    assert hm["balance_scale"] > 1.0
    assert hm["balanced_feature"] > 0.0
    assert hm["balanced_structure"] > 0.0
    assert hm["balanced_ratio"] > 0.9
    assert hm["balanced_ratio"] < 1.1
