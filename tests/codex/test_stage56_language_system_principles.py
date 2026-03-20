from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_language_system_principles.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_language_system_margin_positive() -> None:
    mod = _load_module()
    summary = mod.build_language_system_principles_summary()
    hm = summary["headline_metrics"]
    assert hm["language_entry_core"] > hm["language_pressure_core"]
    assert hm["language_system_margin"] > 0.0
