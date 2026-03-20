from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_encoding_stage_summary import build_encoding_stage_summary


def test_encoding_stage_summary_positive() -> None:
    summary = build_encoding_stage_summary()
    hm = summary["headline_metrics"]

    assert hm["margin_v17_to_v21_mean"] > 0.0
    assert 0.0 < hm["convergence_smoothness"] <= 1.0
    assert hm["feature_structure_ratio"] > 0.0
    assert hm["learning_pressure_ratio"] > 1.0
    assert hm["stage_balance"] > 0.0
