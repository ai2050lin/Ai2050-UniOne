from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_language_total_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_language_total_margin_positive() -> None:
    mod = _load_module()
    summary = mod.build_language_total_analysis_summary()
    hm = summary["headline_metrics"]
    assert hm["language_total_margin"] > 0.0
    assert hm["language_remaining_gap"] >= 0.0

