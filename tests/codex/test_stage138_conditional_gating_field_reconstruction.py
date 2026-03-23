#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage138_conditional_gating_field_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("stage138_conditional_gating_field_reconstruction", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage138_conditional_gating_field_reconstruction() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["family_count"] == 5
    assert summary["best_law_name"] == "qbg_linear_grid"
    assert summary["best_weights"]["q"] == 0.1
    assert summary["best_weights"]["b"] == 0.1
    assert summary["best_weights"]["g"] == 0.8
    assert summary["best_correlation"] > 0.83
    assert summary["best_mae"] < 0.03
    assert summary["conditional_gating_field_score"] > 0.93
    assert summary["weakest_proxy_name"] == "b_proxy_mean"
    assert summary["strongest_proxy_name"] == "q_proxy_mean"

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE138_CONDITIONAL_GATING_FIELD_RECONSTRUCTION_REPORT.md"
    family_path = OUTPUT_DIR / "family_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert family_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_conditional_gating_field_ready"


if __name__ == "__main__":
    test_stage138_conditional_gating_field_reconstruction()
    print("test_stage138_conditional_gating_field_reconstruction: PASS")
