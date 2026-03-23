#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage129_noun_neuron_rule_compression.py"
SPEC = importlib.util.spec_from_file_location("stage129_noun_neuron_rule_compression", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage129_noun_neuron_rule_compression() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["feature_row_count"] >= 10
    assert summary["best_law_name"] == "linear_mean"
    assert summary["best_law_score"] > 0.75
    assert summary["best_law_correlation"] > 0.75
    assert summary["noun_neuron_rule_compression_score"] > 0.70
    assert len(summary["law_rows"]) >= 4
    assert summary["top_compressed_neurons"][0]["layer_index"] == 11

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE129_NOUN_NEURON_RULE_COMPRESSION_REPORT.md"
    law_path = OUTPUT_DIR / "law_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert law_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_noun_neuron_rule_compression_ready"


if __name__ == "__main__":
    test_stage129_noun_neuron_rule_compression()
    print("test_stage129_noun_neuron_rule_compression: PASS")
