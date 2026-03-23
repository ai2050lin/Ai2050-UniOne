#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage134_noun_verb_joint_propagation.py"
SPEC = importlib.util.spec_from_file_location("stage134_noun_verb_joint_propagation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage134_noun_verb_joint_propagation() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["family_count"] == 5
    assert summary["noun_sample_count"] == 320
    assert summary["verb_count"] == 6
    assert summary["case_count_per_family"] == 1920
    assert summary["mean_noun_route_corr"] > 0.10
    assert summary["mean_sign_consistency_rate"] > 0.50
    assert summary["mean_route_band_gap"] > 0.008
    assert summary["positive_family_rate"] == 1.0
    assert summary["noun_verb_joint_propagation_score"] > 0.54

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE134_NOUN_VERB_JOINT_PROPAGATION_REPORT.md"
    family_path = OUTPUT_DIR / "family_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert family_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_noun_verb_joint_ready"


if __name__ == "__main__":
    test_stage134_noun_verb_joint_propagation()
    print("test_stage134_noun_verb_joint_propagation: PASS")
