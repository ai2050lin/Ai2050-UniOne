#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage140_deepseek_language_validation_suite import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=False)
    assert summary["status_short"] == "deepseek7b_language_validation_ready"
    assert summary["model_name"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert summary["vocab_summary"]["clean_unique_word_count"] > 10000
    assert summary["field_summary"]["conditional_gating_field_score"] >= 0.0
    assert summary["transfer_summary"]["transfer_verdict"] in {
        "theory_transfer_strong",
        "theory_transfer_partial",
        "theory_transfer_weak",
    }
    assert (Path(OUTPUT_DIR) / "summary.json").exists()
    assert (Path(OUTPUT_DIR) / "STAGE140_DEEPSEEK_LANGUAGE_VALIDATION_SUITE_REPORT.md").exists()
    print("PASS")


if __name__ == "__main__":
    main()
