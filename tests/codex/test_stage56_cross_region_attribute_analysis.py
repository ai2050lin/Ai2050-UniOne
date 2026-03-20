from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_cross_region_attribute_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_attributes_are_distributed() -> None:
    mod = _load_module()
    summary = mod.build_cross_region_attribute_summary()
    hm = summary["headline_metrics"]
    assert hm["attribute_distributed_score"] > 0.4
    assert hm["attribute_single_region_score"] < hm["attribute_anchor_mass"]
