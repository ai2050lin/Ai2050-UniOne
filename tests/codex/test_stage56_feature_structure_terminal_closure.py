from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_feature_structure_terminal_closure import build_feature_structure_terminal_closure_summary


def test_feature_structure_terminal_closure_positive() -> None:
    summary = build_feature_structure_terminal_closure_summary()
    hm = summary["headline_metrics"]

    assert hm["terminal_circuit_closure"] > 0.0
    assert hm["terminal_structure_closure"] > hm["terminal_circuit_closure"]
    assert hm["terminal_feedback_closure"] > 0.0
    assert hm["terminal_closure_margin_v3"] > hm["terminal_structure_closure"]
