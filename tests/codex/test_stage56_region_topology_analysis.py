from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_region_topology_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_region_topology_is_positive() -> None:
    mod = _load_module()
    summary = mod.build_region_topology_summary()
    hm = summary["headline_metrics"]
    assert hm["family_region_density"] > 0.8
    assert hm["region_topology_margin"] > 0.0
