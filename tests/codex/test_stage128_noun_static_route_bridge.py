#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage128_noun_static_route_bridge.py"
SPEC = importlib.util.spec_from_file_location("stage128_noun_static_route_bridge", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage128_noun_static_route_bridge() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["noun_layer_index"] == 11
    assert summary["route_layer_index"] == 3
    assert summary["noun_selected_count"] == 16
    assert summary["route_selected_count"] == 16
    assert summary["bridge_alignment_mean"] > 0.05
    assert summary["route_alignment_mean"] > 0.05
    assert summary["positive_bridge_rate"] >= 0.25
    assert summary["noun_static_route_bridge_score"] > 0.30
    assert summary["strongest_bridge_pairs"][0]["cosine_alignment"] > 0.10

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE128_NOUN_STATIC_ROUTE_BRIDGE_REPORT.md"
    pairs_path = OUTPUT_DIR / "strongest_bridge_pairs.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert pairs_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_noun_static_route_bridge_ready"


if __name__ == "__main__":
    test_stage128_noun_static_route_bridge()
    print("test_stage128_noun_static_route_bridge: PASS")
