from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_spiking_network_path_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_language_unlock_is_not_direct_agi_unlock() -> None:
    mod = _load_module()
    summary = mod.build_spiking_network_path_summary()
    hm = summary["headline_metrics"]
    assert hm["feature_extraction_unlock"] > hm["direct_agi_unlock"]
    assert hm["overlinearity_penalty"] > 0.0
