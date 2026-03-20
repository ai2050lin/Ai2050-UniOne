from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_encoding_feature_principles import build_encoding_feature_principles_summary


def test_encoding_feature_principles_positive() -> None:
    summary = build_encoding_feature_principles_summary()
    hm = summary["headline_metrics"]

    assert hm["extraction_stack"] > 0.0
    assert hm["structure_stack"] > hm["extraction_stack"]
    assert hm["equalized_core"] > 0.0
    assert hm["principle_margin"] > 1.0
    assert len(summary["principles"]) >= 4
