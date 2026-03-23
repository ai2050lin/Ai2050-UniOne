#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage120_lexical_type_projection_atlas.py"
SPEC = importlib.util.spec_from_file_location("stage120_lexical_type_projection_atlas", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage120_lexical_type_projection_atlas() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["bridge_type_name"] == "adverb"
    assert 0.70 < summary["lexical_type_projection_atlas_score"] < 0.90
    assert summary["noun_meso_anchor_ratio"] > 0.75
    assert summary["verb_macro_anchor_ratio"] > 0.90
    assert summary["adjective_micro_anchor_ratio"] > 0.50
    assert summary["function_macro_anchor_ratio"] > 0.65
    assert summary["adverb_bridge_entropy"] > 0.85

    type_rows = {row["lexical_type"]: row for row in summary["type_rows"]}
    assert type_rows["noun"]["dominant_band"] == "meso"
    assert type_rows["verb"]["dominant_band"] == "macro"
    assert type_rows["adjective"]["dominant_band"] == "micro"
    assert type_rows["function"]["dominant_band"] == "macro"

    assert len(summary["boundary_words"]["noun"]) >= 5
    assert len(summary["boundary_words"]["adverb"]) >= 5
    assert len(summary["drift_words"]["adjective"]) >= 3

    summary_path = OUTPUT_DIR / "summary.json"
    csv_path = OUTPUT_DIR / "lexical_type_rows.csv"
    report_path = OUTPUT_DIR / "STAGE120_LEXICAL_TYPE_PROJECTION_ATLAS_REPORT.md"
    assert summary_path.exists()
    assert csv_path.exists()
    assert report_path.exists()

    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded_summary["status_short"] == "gpt2_lexical_type_projection_atlas_ready"


if __name__ == "__main__":
    test_stage120_lexical_type_projection_atlas()
    print("test_stage120_lexical_type_projection_atlas: PASS")
