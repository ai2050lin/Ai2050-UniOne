#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage139_qwen3_language_validation_suite import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=False)

    assert summary["status_short"] == "qwen3_language_validation_ready"
    assert summary["vocab_summary"]["clean_unique_word_count"] > 10000
    assert summary["adverb_summary"]["adverb_gate_bridge_score"] >= 0.0
    assert summary["dynamic_summary"]["adverb_context_route_shift_score"] >= 0.0
    assert summary["route_summary"]["route_shift_layer_localization_score"] >= 0.0
    assert summary["noun_basic_summary"]["dominant_general_layer_index"] >= 0
    assert summary["noun_context_summary"]["syntax_stability_rate"] >= 0.0
    assert summary["discourse_summary"]["complex_discourse_noun_propagation_score"] >= 0.0
    assert summary["joint_summary"]["noun_verb_joint_propagation_score"] >= 0.0
    assert summary["anaphora_summary"]["anaphora_ellipsis_score"] >= 0.0
    assert summary["result_summary"]["noun_verb_result_chain_score"] >= 0.0
    assert summary["field_summary"]["conditional_gating_field_score"] >= 0.0
    assert summary["transfer_summary"]["transfer_verdict"] in {
        "theory_transfer_strong",
        "theory_transfer_partial",
        "theory_transfer_weak",
    }
    assert (Path(OUTPUT_DIR) / "summary.json").exists()
    assert (Path(OUTPUT_DIR) / "STAGE139_QWEN3_LANGUAGE_VALIDATION_SUITE_REPORT.md").exists()
    print("PASS")


if __name__ == "__main__":
    main()
