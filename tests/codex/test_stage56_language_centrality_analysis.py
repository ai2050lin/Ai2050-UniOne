from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_language_centrality_analysis.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_language_is_highly_central_but_not_total() -> None:
    mod = _load_module()
    summary = mod.build_language_centrality_summary()
    hm = summary["headline_metrics"]
    assert hm["language_centrality"] > 0.8
    assert hm["language_residual"] > 0.0
