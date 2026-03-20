from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_sparse_activation_region_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sparse_activation_efficiency_positive() -> None:
    mod = _load_module()
    summary = mod.build_sparse_activation_region_summary()
    hm = summary["headline_metrics"]
    assert hm["sparse_activation_efficiency"] > 0.3
    assert hm["sparse_seed_activation"] > hm["sparse_structure_activation"] / 2
