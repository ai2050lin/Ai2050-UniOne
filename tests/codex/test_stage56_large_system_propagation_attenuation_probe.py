from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_large_system_propagation_attenuation_probe.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_large_system_propagation_attenuation_probe_positive() -> None:
    mod = _load_module()
    summary = mod.build_large_system_propagation_attenuation_probe_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["attenuation_structure"] <= 1.0
    assert 0.0 <= hm["attenuation_context"] <= 1.0
    assert 0.0 <= hm["attenuation_route"] <= 1.0
    assert 0.0 <= hm["attenuation_learning"] <= 1.0
    assert 0.0 <= hm["anti_attenuation_readiness"] <= 1.0
    assert hm["anti_attenuation_margin"] > 0.0


if __name__ == "__main__":
    test_large_system_propagation_attenuation_probe_positive()
