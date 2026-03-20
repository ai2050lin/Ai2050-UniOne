from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_object_attribute_coupling_prototype.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_object_attribute_prototype_shared_then_split() -> None:
    mod = _load_module()
    summary = mod.build_object_attribute_coupling_summary(steps=120)
    hm = summary["headline_metrics"]
    assert hm["shared_attribute_reuse"] > 0.5
    assert hm["same_attribute_different_route"] > 0.0

