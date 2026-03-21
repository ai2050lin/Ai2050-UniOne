from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_large_scale_high_intensity_long_horizon_extreme.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_large_scale_high_intensity_long_horizon_extreme_positive() -> None:
    mod = _load_module()
    summary = mod.build_large_scale_high_intensity_long_horizon_extreme_summary()
    hm = summary["headline_metrics"]
    assert hm["extreme_language_keep"] > 0.0
    assert hm["extreme_structure_keep"] > 0.0
    assert hm["extreme_context_keep"] > 0.0
    assert 0.0 <= hm["extreme_forgetting_penalty"] <= 1.0
    assert 0.0 <= hm["extreme_collapse_risk"] <= 1.0


if __name__ == "__main__":
    test_large_scale_high_intensity_long_horizon_extreme_positive()
