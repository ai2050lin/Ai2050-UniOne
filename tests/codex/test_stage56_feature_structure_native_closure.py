from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_feature_structure_native_closure import build_feature_structure_native_closure_summary


def test_feature_structure_native_closure_positive() -> None:
    summary = build_feature_structure_native_closure_summary()
    hm = summary["headline_metrics"]

    assert hm["closure_circuit_link"] > 0.0
    assert hm["closure_structure_link"] > hm["closure_circuit_link"]
    assert hm["closure_feedback"] > 0.0
    assert hm["native_closure_margin"] > hm["closure_structure_link"]
