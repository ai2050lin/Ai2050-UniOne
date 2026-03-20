from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_structure_genesis_confidence_stabilization import (
    build_structure_genesis_confidence_stabilization_summary,
)


def test_structure_genesis_confidence_stabilization_positive() -> None:
    summary = build_structure_genesis_confidence_stabilization_summary()
    hm = summary["headline_metrics"]

    assert hm["stabilized_branching"] > 0.0
    assert hm["stabilized_binding"] > hm["stabilized_branching"]
    assert hm["stabilized_feedback"] > 0.0
    assert hm["stabilized_margin"] > hm["stabilized_binding"]
    assert hm["stabilized_confidence"] > 0.0
