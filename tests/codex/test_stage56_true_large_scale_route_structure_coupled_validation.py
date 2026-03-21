from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_true_large_scale_route_structure_coupled_validation.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_true_large_scale_route_structure_coupled_validation_positive() -> None:
    mod = _load_module()
    summary = mod.build_true_large_scale_route_structure_coupled_validation_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["coupled_route_keep"] <= 1.0
    assert 0.0 <= hm["coupled_structure_keep"] <= 1.0
    assert 0.0 <= hm["coupled_context_keep"] <= 1.0
    assert 0.0 <= hm["coupled_failure_risk"] <= 1.0
    assert 0.0 <= hm["coupled_readiness"] <= 1.0


if __name__ == "__main__":
    test_true_large_scale_route_structure_coupled_validation_positive()
