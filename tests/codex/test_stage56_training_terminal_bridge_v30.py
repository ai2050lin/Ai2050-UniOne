from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests" / "codex" / "stage56_training_terminal_bridge_v30.py"
SPEC = importlib.util.spec_from_file_location("stage56_training_terminal_bridge_v30", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_training_terminal_bridge_v30_summary = MODULE.build_training_terminal_bridge_v30_summary


def test_stage56_training_terminal_bridge_v30_metrics() -> None:
    summary = build_training_terminal_bridge_v30_summary()
    hm = summary["headline_metrics"]

    assert 0.0 <= hm["plasticity_rule_alignment_v30"] <= 1.0
    assert 0.0 <= hm["structure_rule_alignment_v30"] <= 1.0
    assert 0.0 <= hm["topology_training_readiness_v30"] <= 1.0
    assert 0.0 <= hm["topology_training_gap_v30"] <= 1.0
    assert 0.0 <= hm["systemic_steady_guard_v30"] <= 1.0
