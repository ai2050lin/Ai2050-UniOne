#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage121_adverb_gate_bridge_probe.py"
SPEC = importlib.util.spec_from_file_location("stage121_adverb_gate_bridge_probe", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage121_adverb_gate_bridge_probe() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    type_means = summary["type_means"]
    assert summary["core_adverb_count"] > 300
    assert summary["adverb_gate_bridge_score"] > 0.45
    assert 0.10 < summary["adverb_midpoint_position"] < 0.60
    assert summary["adverb_action_function_balance_mean"] > 0.88

    assert type_means["noun"]["mean_gate_bridge_score"] < summary["adverb_gate_mean"]
    assert type_means["adjective"]["mean_gate_bridge_score"] < summary["adverb_gate_mean"]
    assert summary["adverb_gate_mean"] < summary["control_gate_mean"]

    assert summary["prototype_seed_counts"]["verb"] > 200
    assert summary["prototype_seed_counts"]["function"] > 50
    assert summary["prototype_seed_counts"]["noun"] > 1000
    assert summary["prototype_seed_counts"]["adjective"] > 100

    assert len(summary["top_gate_adverbs"]) >= 10
    top_words = {row["word"] for row in summary["top_gate_adverbs"][:12]}
    assert "also" in top_words or "actually" in top_words or "therefore" in top_words

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE121_ADVERB_GATE_BRIDGE_PROBE_REPORT.md"
    csv_path = OUTPUT_DIR / "top_gate_adverbs.csv"
    assert summary_path.exists()
    assert report_path.exists()
    assert csv_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_adverb_gate_bridge_probe_ready"


if __name__ == "__main__":
    test_stage121_adverb_gate_bridge_probe()
    print("test_stage121_adverb_gate_bridge_probe: PASS")
