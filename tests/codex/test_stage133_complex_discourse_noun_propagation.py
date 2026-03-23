#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage133_complex_discourse_noun_propagation.py"
SPEC = importlib.util.spec_from_file_location("stage133_complex_discourse_noun_propagation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage133_complex_discourse_noun_propagation() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["family_count"] == 5
    assert summary["sample_count"] == 2048
    assert summary["early_layer_index"] == 1
    assert summary["late_layer_index"] == 11
    assert summary["mean_early_remention_corr"] > 0.95
    assert summary["mean_late_remention_corr"] > 0.60
    assert summary["early_positive_family_rate"] == 1.0
    assert summary["late_positive_family_rate"] == 1.0
    assert summary["complex_discourse_noun_propagation_score"] > 0.90

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE133_COMPLEX_DISCOURSE_NOUN_PROPAGATION_REPORT.md"
    family_path = OUTPUT_DIR / "family_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert family_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_complex_discourse_noun_ready"


if __name__ == "__main__":
    test_stage133_complex_discourse_noun_propagation()
    print("test_stage133_complex_discourse_noun_propagation: PASS")
