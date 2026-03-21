from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests" / "codex" / "stage56_large_system_stable_amplification_strengthening.py"
SPEC = importlib.util.spec_from_file_location("stage56_large_system_stable_amplification_strengthening", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_large_system_stable_amplification_strengthening_summary = MODULE.build_large_system_stable_amplification_strengthening_summary


def test_stage56_large_system_stable_amplification_strengthening_metrics() -> None:
    summary = build_large_system_stable_amplification_strengthening_summary()
    hm = summary["headline_metrics"]

    assert 0.0 <= hm["stable_reinforced_strength"] <= 1.0
    assert 0.0 <= hm["stable_reinforced_structure"] <= 1.0
    assert 0.0 <= hm["stable_reinforced_route"] <= 1.0
    assert 0.0 <= hm["stable_reinforced_learning"] <= 1.0
    assert 0.0 <= hm["stable_reinforced_penalty"] <= 1.0
    assert 0.0 <= hm["stable_reinforced_readiness"] <= 1.0
    assert 0.0 <= hm["stable_reinforced_score"] <= 1.0
    assert hm["stable_reinforced_margin"] > 0.0
