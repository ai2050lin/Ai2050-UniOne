from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_network_structure_genesis_probe import build_network_structure_genesis_probe_summary


def test_network_structure_genesis_probe_positive() -> None:
    summary = build_network_structure_genesis_probe_summary()
    hm = summary["headline_metrics"]

    assert hm["feature_to_structure_gain"] > 0.0
    assert hm["circuit_binding_gain"] > 0.0
    assert hm["feedback_retention"] > 0.0
    assert hm["genesis_margin"] > hm["feature_to_structure_gain"]
