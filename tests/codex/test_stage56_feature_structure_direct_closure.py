from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_feature_structure_direct_closure import build_feature_structure_direct_closure_summary


def test_feature_structure_direct_closure_positive() -> None:
    summary = build_feature_structure_direct_closure_summary()
    hm = summary["headline_metrics"]

    assert hm["direct_circuit_closure"] > 0.0
    assert hm["direct_structure_closure"] > hm["direct_circuit_closure"]
    assert hm["direct_feedback_closure"] > 0.0
    assert hm["direct_closure_margin_v2"] > hm["direct_structure_closure"]
