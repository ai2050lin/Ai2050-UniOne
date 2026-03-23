#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage122_adverb_context_route_shift_probe.py"
SPEC = importlib.util.spec_from_file_location("stage122_adverb_context_route_shift_probe", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage122_adverb_context_route_shift_probe() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["case_count"] == 144
    assert summary["adverb_verb_route_delta_mean"] > summary["adjective_verb_route_delta_mean"]
    assert summary["verb_route_advantage_mean"] > 0.0002
    assert summary["verb_route_peak_advantage_mean"] > 0.003
    assert summary["modifier_attention_advantage_mean"] > 0.02
    assert summary["positive_peak_route_shift_case_rate"] > 0.70
    assert summary["adverb_context_route_shift_score"] > 0.45

    best_words = {row["adverb"] for row in summary["best_route_shift_cases"][:20]}
    assert "actually" in best_words or "probably" in best_words or "therefore" in best_words

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE122_ADVERB_CONTEXT_ROUTE_SHIFT_PROBE_REPORT.md"
    cases_path = OUTPUT_DIR / "best_route_shift_cases.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert cases_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_adverb_context_route_shift_ready"


if __name__ == "__main__":
    test_stage122_adverb_context_route_shift_probe()
    print("test_stage122_adverb_context_route_shift_probe: PASS")
