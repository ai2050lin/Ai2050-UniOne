from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_feature_layer_terminal_direct import build_feature_layer_terminal_direct_summary


def test_feature_layer_terminal_direct_positive() -> None:
    summary = build_feature_layer_terminal_direct_summary()
    hm = summary["headline_metrics"]

    assert hm["direct_basis_v5"] > 0.0
    assert hm["direct_selectivity_v5"] > hm["direct_basis_v5"] / 5.0
    assert hm["direct_lock_v5"] > hm["direct_selectivity_v5"]
    assert hm["feature_terminal_core_v5"] > hm["direct_lock_v5"]
