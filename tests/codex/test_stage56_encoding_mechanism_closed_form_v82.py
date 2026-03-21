from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests" / "codex" / "stage56_encoding_mechanism_closed_form_v82.py"
SPEC = importlib.util.spec_from_file_location("stage56_encoding_mechanism_closed_form_v82", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_encoding_mechanism_closed_form_v82_summary = MODULE.build_encoding_mechanism_closed_form_v82_summary


def test_stage56_encoding_mechanism_closed_form_v82_metrics() -> None:
    summary = build_encoding_mechanism_closed_form_v82_summary()
    hm = summary["headline_metrics"]

    assert hm["feature_term_v82"] > 0.0
    assert hm["structure_term_v82"] > 0.0
    assert hm["learning_term_v82"] > 0.0
    assert hm["pressure_term_v82"] >= 0.0
    assert hm["encoding_margin_v82"] > hm["feature_term_v82"]
