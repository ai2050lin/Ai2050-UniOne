from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_language_gap_bottleneck_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_language_gap_not_primary_bottleneck() -> None:
    mod = _load_module()
    summary = mod.build_language_gap_bottleneck_summary()
    hm = summary["headline_metrics"]
    assert hm["language_gap_remaining"] < hm["math_unification_gap"]
    assert hm["language_gap_remaining"] < hm["agi_realization_gap"]
    assert hm["language_is_primary_bottleneck"] == 0.0
