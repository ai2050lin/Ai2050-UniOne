#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage136_anaphora_ellipsis_propagation.py"
SPEC = importlib.util.spec_from_file_location("stage136_anaphora_ellipsis_propagation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage136_anaphora_ellipsis_propagation() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["family_count"] == 5
    assert summary["sample_count"] == 2048
    assert summary["mean_noun_pronoun_early_corr"] < 0.0
    assert summary["mean_noun_ellipsis_early_corr"] < 0.0
    assert summary["mean_noun_pronoun_late_corr"] > 0.10
    assert summary["mean_noun_ellipsis_late_corr"] > 0.35
    assert summary["mean_pronoun_sign_consistency_rate"] > 0.95
    assert summary["anaphora_ellipsis_propagation_score"] > 0.56

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE136_ANAPHORA_ELLIPSIS_PROPAGATION_REPORT.md"
    family_path = OUTPUT_DIR / "family_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert family_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_anaphora_ellipsis_ready"


if __name__ == "__main__":
    test_stage136_anaphora_ellipsis_propagation()
    print("test_stage136_anaphora_ellipsis_propagation: PASS")
