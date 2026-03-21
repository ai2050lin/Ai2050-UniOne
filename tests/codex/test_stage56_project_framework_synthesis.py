from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_project_framework_synthesis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_project_framework_synthesis_positive() -> None:
    mod = _load_module()
    summary = mod.build_project_framework_synthesis_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["language_anchor"] <= 1.0
    assert 0.0 <= hm["language_to_brain_bridge"] <= 1.0
    assert 0.0 <= hm["brain_to_topology_bridge"] <= 1.0
    assert 0.0 <= hm["topology_to_training_bridge"] <= 1.0
    assert hm["framework_synthesis_margin"] > 0.0


if __name__ == "__main__":
    test_project_framework_synthesis_positive()
