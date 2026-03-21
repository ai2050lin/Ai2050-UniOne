from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_large_system_steady_amplification_validation.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_large_system_steady_amplification_validation_positive() -> None:
    mod = _load_module()
    summary = mod.build_large_system_steady_amplification_validation_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["steady_amplification_strength"] <= 1.0
    assert 0.0 <= hm["steady_structure_stability"] <= 1.0
    assert 0.0 <= hm["steady_route_stability"] <= 1.0
    assert 0.0 <= hm["steady_learning_lift"] <= 1.0
    assert 0.0 <= hm["steady_readiness"] <= 1.0
    assert hm["steady_margin"] > 0.0


if __name__ == "__main__":
    test_large_system_steady_amplification_validation_positive()
