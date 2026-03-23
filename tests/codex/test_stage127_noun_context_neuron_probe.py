#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage127_noun_context_neuron_probe.py"
SPEC = importlib.util.spec_from_file_location("stage127_noun_context_neuron_probe", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage127_noun_context_neuron_probe() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["target_lexical_type"] == "noun"
    assert summary["target_count"] == 27702
    assert summary["control_count"] == 2839
    assert summary["layer_count"] == 12
    assert summary["neurons_per_layer"] == 3072
    assert summary["context_template_count"] == 3
    assert summary["stage124_dominant_general_layer_index"] == 11
    assert summary["dominant_general_layer_index"] in {0, 1}
    assert summary["l11_rule_preserved"] is False
    assert summary["noun_context_neuron_probe_score"] >= 0.50
    assert summary["top_general_neurons"][0]["general_rule_score"] > 0.65

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE127_NOUN_CONTEXT_NEURON_PROBE_REPORT.md"
    general_path = OUTPUT_DIR / "top_general_neurons.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert general_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_noun_context_neuron_probe_ready"


if __name__ == "__main__":
    test_stage127_noun_context_neuron_probe()
    print("test_stage127_noun_context_neuron_probe: PASS")
