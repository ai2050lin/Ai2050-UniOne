#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage137_noun_verb_result_chain.py"
SPEC = importlib.util.spec_from_file_location("stage137_noun_verb_result_chain", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage137_noun_verb_result_chain() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["family_count"] == 5
    assert summary["noun_sample_count"] == 320
    assert summary["verb_count"] == 6
    assert summary["case_count_per_family"] == 1920
    assert summary["mean_noun_verb_corr"] > 0.10
    assert summary["mean_verb_result_corr"] > 0.08
    assert summary["mean_noun_result_corr"] > 0.05
    assert summary["positive_bridge_rate"] >= 0.8
    assert summary["noun_verb_result_chain_score"] > 0.56

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE137_NOUN_VERB_RESULT_CHAIN_REPORT.md"
    family_path = OUTPUT_DIR / "family_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert family_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_noun_verb_result_ready"


if __name__ == "__main__":
    test_stage137_noun_verb_result_chain()
    print("test_stage137_noun_verb_result_chain: PASS")
