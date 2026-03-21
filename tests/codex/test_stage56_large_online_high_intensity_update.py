from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_large_online_high_intensity_update.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_large_online_high_intensity_update_positive() -> None:
    mod = _load_module()
    summary = mod.build_large_online_high_intensity_update_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["high_intensity_language_keep"] <= 1.0
    assert 0.0 <= hm["high_intensity_novel_gain"] <= 1.0
    assert 0.0 <= hm["high_intensity_structure_keep"] <= 1.0
    assert 0.0 <= hm["high_intensity_stability"] <= 1.0


if __name__ == "__main__":
    test_large_online_high_intensity_update_positive()
