from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_encoding_mechanism_closed_form_v51.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v51_margin_positive() -> None:
    mod = _load_module()
    summary = mod.build_encoding_mechanism_closed_form_v51_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_margin_v51"] > hm["learning_term_v51"]
    assert hm["pressure_term_v51"] >= 0.0
