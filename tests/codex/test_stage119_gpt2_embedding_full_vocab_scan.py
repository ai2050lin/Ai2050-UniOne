#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage119_gpt2_embedding_full_vocab_scan.py"
SPEC = importlib.util.spec_from_file_location("stage119_gpt2_embedding_full_vocab_scan", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
run_analysis = MODULE.run_analysis


def test_stage119_gpt2_embedding_full_vocab_scan() -> None:
    summary, rows = run_analysis(output_dir=OUTPUT_DIR)

    assert summary["vocab_size"] == 50257
    assert summary["embedding_dim"] == 768
    assert summary["clean_token_variant_count"] > 45000
    assert summary["clean_unique_word_count"] > 30000
    assert summary["seed_group_count"] >= 15
    assert summary["seed_group_coverage"] > 0.80
    assert summary["lexical_type_group_count"] == 5
    assert summary["lexical_type_coverage"] > 0.84
    assert 0.35 < summary["mean_effective_encoding_score"] < 0.75

    row_map = {row["word"]: row for row in rows}
    apple = row_map["apple"]
    fruit = row_map["fruit"]
    justice = row_map["justice"]
    red = row_map["red"]
    run = row_map["run"]
    build = row_map["build"]
    beautiful = row_map["beautiful"]
    quickly = row_map["quickly"]
    and_word = row_map["and"]

    assert apple["band"] == "meso"
    assert apple["group"] == "meso_fruit"
    assert apple["group_score"] > 0.20

    assert fruit["band"] == "meso"
    assert fruit["group"] == "meso_fruit"

    assert justice["band"] == "macro"
    assert justice["group"] in {"macro_abstract", "macro_system"}
    assert justice["group_score"] > 0.15

    assert red["band"] == "micro"
    assert red["group"] == "micro_color"
    assert red["group_score"] > 0.15

    assert run["band"] == "macro"
    assert run["group"] == "macro_action"
    assert run["lexical_type"] == "verb"

    assert apple["lexical_type"] == "noun"
    assert build["lexical_type"] == "verb"
    assert beautiful["lexical_type"] == "adjective"
    assert quickly["lexical_type"] == "adverb"
    assert and_word["lexical_type"] == "function"

    lexical_type_counts = summary["lexical_type_counts"]
    assert lexical_type_counts["noun"] > 10000
    assert lexical_type_counts["verb"] > 800
    assert lexical_type_counts["adjective"] > 1000
    assert lexical_type_counts["adverb"] > 300
    assert lexical_type_counts["function"] > 300

    summary_path = OUTPUT_DIR / "summary.json"
    csv_path = OUTPUT_DIR / "word_rows.csv"
    jsonl_path = OUTPUT_DIR / "word_rows.jsonl"
    report_path = OUTPUT_DIR / "STAGE119_GPT2_EMBEDDING_FULL_VOCAB_SCAN_REPORT.md"

    assert summary_path.exists()
    assert csv_path.exists()
    assert jsonl_path.exists()
    assert report_path.exists()

    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    assert loaded_summary["status_short"] == "gpt2_embedding_vocab_scan_ready"


if __name__ == "__main__":
    test_stage119_gpt2_embedding_full_vocab_scan()
    print("test_stage119_gpt2_embedding_full_vocab_scan: PASS")
