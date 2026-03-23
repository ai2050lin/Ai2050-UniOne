#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage131_l1_l3_l11_propagation_bridge.py"
SPEC = importlib.util.spec_from_file_location("stage131_l1_l3_l11_propagation_bridge", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage131_l1_l3_l11_propagation_bridge() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["family_count"] == 6
    assert summary["sample_count_per_family"] == 2048
    assert summary["early_layer_index"] == 1
    assert summary["route_layer_index"] == 3
    assert summary["late_layer_index"] == 11
    assert summary["mean_l1_l3_corr"] < 0.0
    assert summary["mean_l3_l11_corr"] < 0.0
    assert summary["mean_l1_l11_corr"] > 0.30
    assert summary["coherent_family_rate"] == 0.0
    assert summary["l1_l3_l11_propagation_score"] > 0.40

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE131_L1_L3_L11_PROPAGATION_BRIDGE_REPORT.md"
    family_path = OUTPUT_DIR / "family_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert family_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_l1_l3_l11_propagation_ready"


if __name__ == "__main__":
    test_stage131_l1_l3_l11_propagation_bridge()
    print("test_stage131_l1_l3_l11_propagation_bridge: PASS")
