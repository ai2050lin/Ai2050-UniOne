from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_color_pathway_overlap_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_color_pathway_overlap_shape() -> None:
    mod = _load_module()
    summary = mod.build_color_pathway_overlap_summary()
    hm = summary["headline_metrics"]
    assert hm["shared_fiber_score"] > hm["same_full_route_score"]
    assert hm["contextual_split_score"] > 0.0

