#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage130_multisyntax_noun_context_probe.py"
SPEC = importlib.util.spec_from_file_location("stage130_multisyntax_noun_context_probe", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage130_multisyntax_noun_context_probe() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["target_lexical_type"] == "noun"
    assert summary["target_count"] == 27702
    assert summary["family_count"] == 6
    assert summary["early_family_count"] == 6
    assert summary["l1_family_count"] >= 3
    assert summary["syntax_stability_rate"] == 1.0
    assert summary["multisyntax_noun_context_score"] > 0.90
    assert summary["dominant_general_layer_index"] == 1
    assert summary["recurrent_early_neurons"][0]["syntax_hit_count"] >= 3

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE130_MULTISYNTAX_NOUN_CONTEXT_PROBE_REPORT.md"
    family_path = OUTPUT_DIR / "family_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert family_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_multisyntax_noun_context_ready"


if __name__ == "__main__":
    test_stage130_multisyntax_noun_context_probe()
    print("test_stage130_multisyntax_noun_context_probe: PASS")
