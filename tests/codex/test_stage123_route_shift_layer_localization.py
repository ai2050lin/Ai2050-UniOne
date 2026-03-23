#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage123_route_shift_layer_localization.py"
SPEC = importlib.util.spec_from_file_location("stage123_route_shift_layer_localization", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage123_route_shift_layer_localization() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["layer_count"] == 12
    assert 0 <= summary["dominant_layer_index"] < 12
    assert 0 <= summary["dominant_peak_layer_index"] < 12
    assert summary["best_band_name"] in {"early", "middle", "late"}
    assert summary["dominant_layer_route_advantage_mean"] > 0.003
    assert summary["band_separation_margin"] > 0.0002
    assert summary["route_shift_layer_localization_score"] > 0.45
    assert len(summary["top_layers"]) == 5

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE123_ROUTE_SHIFT_LAYER_LOCALIZATION_REPORT.md"
    layers_path = OUTPUT_DIR / "layer_rows.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert layers_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded["status_short"] == "gpt2_route_shift_layer_localized"


if __name__ == "__main__":
    test_stage123_route_shift_layer_localization()
    print("test_stage123_route_shift_layer_localization: PASS")
