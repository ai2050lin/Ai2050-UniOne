from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("stage56_encoding_mechanism_closed_form_v59.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_encoding_mechanism_closed_form_v59_positive() -> None:
    mod = _load_module()
    summary = mod.build_encoding_mechanism_closed_form_v59_summary()
    hm = summary["headline_metrics"]
    assert hm["feature_term_v59"] > 0.0
    assert hm["structure_term_v59"] > 0.0
    assert hm["learning_term_v59"] > 0.0
    assert hm["encoding_margin_v59"] > hm["pressure_term_v59"]


if __name__ == "__main__":
    test_encoding_mechanism_closed_form_v59_positive()
