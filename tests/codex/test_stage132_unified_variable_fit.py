#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage132_unified_variable_fit.py"
SPEC = importlib.util.spec_from_file_location("stage132_unified_variable_fit", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage132_unified_variable_fit() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["family_count"] == 6
    assert summary["best_law_name"] == "aqgf_linear_grid"
    assert summary["best_weights"]["a"] == 0.5
    assert summary["best_weights"]["q"] == 0.1
    assert summary["best_weights"]["g"] == 0.2
    assert summary["best_weights"]["f"] == 0.2
    assert summary["best_correlation"] > 0.99
    assert summary["best_mae"] < 0.02
    assert summary["noun_unified_variable_fit_score"] > 0.99
    assert summary["weakest_proxy_name"] == "g_proxy_mean"

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE132_UNIFIED_VARIABLE_FIT_REPORT.md"
    family_path = OUTPUT_DIR / "family_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert family_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_noun_unified_variable_fit_ready"


if __name__ == "__main__":
    test_stage132_unified_variable_fit()
    print("test_stage132_unified_variable_fit: PASS")
