from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests" / "codex" / "stage56_training_terminal_bridge_v29.py"
SPEC = importlib.util.spec_from_file_location("stage56_training_terminal_bridge_v29", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_training_terminal_bridge_v29_summary = MODULE.build_training_terminal_bridge_v29_summary


def test_stage56_training_terminal_bridge_v29_metrics() -> None:
    summary = build_training_terminal_bridge_v29_summary()
    hm = summary["headline_metrics"]

    assert 0.0 <= hm["plasticity_rule_alignment_v29"] <= 1.0
    assert 0.0 <= hm["structure_rule_alignment_v29"] <= 1.0
    assert 0.0 <= hm["topology_training_readiness_v29"] <= 1.0
    assert 0.0 <= hm["topology_training_gap_v29"] <= 1.0
    assert 0.0 <= hm["systemic_guard_v29"] <= 1.0
