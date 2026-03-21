from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_encoding_mechanism_closed_form_v52.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v52_margin_positive() -> None:
    mod = _load_module()
    summary = mod.build_encoding_mechanism_closed_form_v52_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_margin_v52"] > hm["learning_term_v52"]
    assert hm["pressure_term_v52"] >= 0.0
