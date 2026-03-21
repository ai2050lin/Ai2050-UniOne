from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_spike_3d_long_horizon_prototype.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spike_3d_long_horizon_prototype_positive() -> None:
    mod = _load_module()
    summary = mod.build_spike_3d_long_horizon_prototype_summary(steps=120, rounds=2)
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["topo_long_retention"] <= 1.0
    assert 0.0 <= hm["topo_long_shared_survival"] <= 1.0
    assert 0.0 <= hm["topo_long_structural_survival"] <= 1.0
    assert hm["topo_long_margin"] > 0.0


if __name__ == "__main__":
    test_spike_3d_long_horizon_prototype_positive()
