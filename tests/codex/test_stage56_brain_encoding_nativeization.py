from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_brain_encoding_nativeization.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_brain_native_chain_strength_bounds() -> None:
    mod = _load_module()
    summary = mod.build_brain_encoding_nativeization_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["brain_native_chain_strength"] <= 1.0
    assert hm["brain_native_gap"] >= 0.0

