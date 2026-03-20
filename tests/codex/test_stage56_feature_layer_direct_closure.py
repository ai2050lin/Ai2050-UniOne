from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_feature_layer_direct_closure import build_feature_layer_direct_closure_summary


def test_feature_layer_direct_closure_positive() -> None:
    summary = build_feature_layer_direct_closure_summary()
    hm = summary["headline_metrics"]

    assert hm["direct_basis_v4"] > hm["direct_basis_v3"] if "direct_basis_v3" in hm else hm["direct_basis_v4"] > 0.0
    assert hm["direct_selectivity_v4"] > 0.0
    assert hm["direct_lock_v4"] > 0.0
    assert hm["feature_direct_closure_v4"] > hm["direct_lock_v4"]
