from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_3d_topology_scaling_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_3d_topology_scaling_analysis_positive() -> None:
    mod = _load_module()
    summary = mod.build_3d_topology_scaling_analysis_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["scale_transport_retention"] <= 1.0
    assert 0.0 <= hm["scale_modular_reuse"] <= 1.0
    assert 0.0 <= hm["scale_ready_score"] <= 1.0
    assert 0.0 <= hm["scale_collision_penalty"] <= 1.0


if __name__ == "__main__":
    test_3d_topology_scaling_analysis_positive()
