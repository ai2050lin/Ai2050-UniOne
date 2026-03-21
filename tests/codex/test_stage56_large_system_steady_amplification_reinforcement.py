from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests" / "codex" / "stage56_large_system_steady_amplification_reinforcement.py"
SPEC = importlib.util.spec_from_file_location("stage56_large_system_steady_amplification_reinforcement", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_large_system_steady_amplification_reinforcement_summary = MODULE.build_large_system_steady_amplification_reinforcement_summary


def test_stage56_large_system_steady_amplification_reinforcement_metrics() -> None:
    summary = build_large_system_steady_amplification_reinforcement_summary()
    hm = summary["headline_metrics"]

    assert 0.0 <= hm["steady_reinforcement_strength"] <= 1.0
    assert 0.0 <= hm["steady_reinforcement_structure"] <= 1.0
    assert 0.0 <= hm["steady_reinforcement_route"] <= 1.0
    assert 0.0 <= hm["steady_reinforcement_learning"] <= 1.0
    assert 0.0 <= hm["steady_reinforcement_penalty"] <= 1.0
    assert 0.0 <= hm["steady_reinforcement_readiness"] <= 1.0
    assert 0.0 <= hm["steady_reinforcement_score"] <= 1.0
    assert hm["steady_reinforcement_margin"] > 0.0
