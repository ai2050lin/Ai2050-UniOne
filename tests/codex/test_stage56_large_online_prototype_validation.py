from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_large_online_prototype_validation.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_large_online_prototype_validation_positive() -> None:
    mod = _load_module()
    summary = mod.build_large_online_prototype_validation_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["large_online_fit"] <= 1.0
    assert 0.0 <= hm["large_online_novel_gain"] <= 1.0
    assert 0.0 <= hm["large_online_structure_keep"] <= 1.0
    assert 0.0 <= hm["large_online_readiness"] <= 1.0


if __name__ == "__main__":
    test_large_online_prototype_validation_positive()
