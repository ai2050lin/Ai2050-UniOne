from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_online_learning_long_horizon_stability.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_online_learning_long_horizon_stability_positive() -> None:
    mod = _load_module()
    summary = mod.build_online_learning_long_horizon_stability_summary(base_steps=200, phase_steps=50)
    hm = summary["headline_metrics"]
    assert hm["long_horizon_retention"] > 0.0
    assert hm["long_horizon_plasticity"] >= 0.0
    assert hm["shared_fiber_survival"] >= 0.0


if __name__ == "__main__":
    test_online_learning_long_horizon_stability_positive()
