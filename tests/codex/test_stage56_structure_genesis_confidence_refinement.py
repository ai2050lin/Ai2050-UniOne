from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_structure_genesis_confidence_refinement import (
    build_structure_genesis_confidence_refinement_summary,
)


def test_structure_genesis_confidence_refinement_positive() -> None:
    summary = build_structure_genesis_confidence_refinement_summary()
    hm = summary["headline_metrics"]

    assert hm["branching_refined_v2"] > 0.0
    assert hm["binding_refined_v2"] > 0.0
    assert hm["feedback_refined_v2"] > 0.0
    assert hm["structure_genesis_margin_v3"] > hm["binding_refined_v2"]
    assert 0.0 < hm["structure_direct_confidence_v3"] < 1.0
