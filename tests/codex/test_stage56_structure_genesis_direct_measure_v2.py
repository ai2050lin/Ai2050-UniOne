from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_structure_genesis_direct_measure_v2 import build_structure_genesis_direct_measure_v2_summary


def test_structure_genesis_direct_measure_v2_positive() -> None:
    summary = build_structure_genesis_direct_measure_v2_summary()
    hm = summary["headline_metrics"]

    assert hm["structure_branching_direct"] > 0.0
    assert hm["closure_binding_direct"] > 0.0
    assert hm["feedback_stability_direct"] > 0.0
    assert hm["structure_genesis_direct_core"] > hm["structure_branching_direct"]
    assert 0.0 < hm["structure_direct_confidence"] < 1.0
