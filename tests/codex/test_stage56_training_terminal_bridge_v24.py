from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_training_terminal_bridge_v24.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_training_terminal_bridge_v24_positive() -> None:
    mod = _load_module()
    summary = mod.build_training_terminal_bridge_v24_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["plasticity_rule_alignment_v24"] <= 1.0
    assert 0.0 <= hm["structure_rule_alignment_v24"] <= 1.0
    assert 0.0 <= hm["topology_training_readiness_v24"] <= 1.0
    assert 0.0 <= hm["topology_training_gap_v24"] <= 1.0


if __name__ == "__main__":
    test_training_terminal_bridge_v24_positive()
