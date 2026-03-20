from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_feature_layer_native_direct_measure import build_feature_layer_native_direct_measure_summary


def test_feature_layer_native_direct_measure_positive() -> None:
    summary = build_feature_layer_native_direct_measure_summary()
    hm = summary["headline_metrics"]

    assert hm["direct_basis_v3"] > 0.0
    assert hm["direct_selectivity_v3"] > 0.0
    assert hm["direct_lock_v3"] > hm["direct_selectivity_v3"]
    assert hm["feature_direct_core_v3"] > hm["direct_lock_v3"]
