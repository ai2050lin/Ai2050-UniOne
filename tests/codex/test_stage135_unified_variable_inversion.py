#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage135_unified_variable_inversion.py"
SPEC = importlib.util.spec_from_file_location("stage135_unified_variable_inversion", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage135_unified_variable_inversion() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["family_count"] == 5
    assert summary["best_law_name"] == "arqgfb_linear_grid"
    assert summary["best_weights"]["a"] == 0.4
    assert summary["best_weights"]["r"] == 0.0
    assert summary["best_weights"]["q"] == 0.0
    assert summary["best_weights"]["g"] == 0.1
    assert summary["best_weights"]["f"] == 0.3
    assert summary["best_weights"]["b"] == 0.2
    assert summary["best_correlation"] > 0.85
    assert summary["best_mae"] < 0.03
    assert summary["unified_variable_inversion_score"] > 0.94
    assert summary["weakest_proxy_name"] == "b_proxy_mean"
    assert summary["strongest_proxy_name"] == "r_proxy_mean"

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE135_UNIFIED_VARIABLE_INVERSION_REPORT.md"
    family_path = OUTPUT_DIR / "family_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert family_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_unified_variable_inversion_ready"


if __name__ == "__main__":
    test_stage135_unified_variable_inversion()
    print("test_stage135_unified_variable_inversion: PASS")
