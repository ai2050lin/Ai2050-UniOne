from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests" / "codex" / "stage56_brain_encoding_direct_refinement_v21.py"
SPEC = importlib.util.spec_from_file_location("stage56_brain_encoding_direct_refinement_v21", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_brain_encoding_direct_refinement_v21_summary = MODULE.build_brain_encoding_direct_refinement_v21_summary


def test_stage56_brain_encoding_direct_refinement_v21_metrics() -> None:
    summary = build_brain_encoding_direct_refinement_v21_summary()
    hm = summary["headline_metrics"]

    assert 0.0 <= hm["direct_origin_measure_v21"] <= 1.0
    assert 0.0 <= hm["direct_feature_measure_v21"] <= 1.0
    assert 0.0 <= hm["direct_structure_measure_v21"] <= 1.0
    assert 0.0 <= hm["direct_route_measure_v21"] <= 1.0
    assert 0.0 <= hm["direct_brain_measure_v21"] <= 1.0
    assert 0.0 <= hm["direct_brain_gap_v21"] <= 1.0
    assert 0.0 <= hm["direct_stable_alignment_v21"] <= 1.0
