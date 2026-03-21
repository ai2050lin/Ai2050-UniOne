from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests" / "codex" / "stage56_large_system_low_risk_steady_amplification_validation.py"
SPEC = importlib.util.spec_from_file_location("stage56_large_system_low_risk_steady_amplification_validation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_large_system_low_risk_steady_amplification_validation_summary = MODULE.build_large_system_low_risk_steady_amplification_validation_summary


def test_stage56_large_system_low_risk_steady_amplification_validation_metrics() -> None:
    summary = build_large_system_low_risk_steady_amplification_validation_summary()
    hm = summary["headline_metrics"]

    assert 0.0 <= hm["low_risk_strength"] <= 1.0
    assert 0.0 <= hm["low_risk_structure"] <= 1.0
    assert 0.0 <= hm["low_risk_route"] <= 1.0
    assert 0.0 <= hm["low_risk_learning"] <= 1.0
    assert 0.0 <= hm["low_risk_penalty"] <= 1.0
    assert 0.0 <= hm["low_risk_readiness"] <= 1.0
    assert 0.0 <= hm["low_risk_score"] <= 1.0
    assert hm["low_risk_margin"] > 0.0
