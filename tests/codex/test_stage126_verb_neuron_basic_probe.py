#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage126_verb_neuron_basic_probe.py"
SPEC = importlib.util.spec_from_file_location("stage126_verb_neuron_basic_probe", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage126_verb_neuron_basic_probe() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["target_lexical_type"] == "verb"
    assert summary["target_count"] == 866
    assert summary["control_count"] == 29675
    assert summary["layer_count"] == 12
    assert summary["neurons_per_layer"] == 3072
    assert summary["target_group_count"] >= 4
    assert summary["primary_band_name"] == "macro"
    assert summary["secondary_band_name"] == "meso"
    assert summary["dominant_general_layer_score"] > 0.53
    assert summary["wordclass_neuron_basic_probe_score"] > 0.95
    assert len(summary["top_general_neurons"]) >= 12
    assert summary["top_general_neurons"][0]["general_rule_score"] > 0.70
    assert summary["top_general_neurons"][0]["group_support_ratio"] > 0.90

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE126_VERB_NEURON_BASIC_PROBE_REPORT.md"
    general_path = OUTPUT_DIR / "top_general_neurons.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert general_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_verb_neuron_probe_ready"


if __name__ == "__main__":
    test_stage126_verb_neuron_basic_probe()
    print("test_stage126_verb_neuron_basic_probe: PASS")
