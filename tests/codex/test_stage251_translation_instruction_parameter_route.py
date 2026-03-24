#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage251_translation_instruction_parameter_route import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=False)
    assert summary["instruction_token_count"] > 0
    assert summary["instruction_attention_mid"] > 0.1
    assert summary["top_gate_shift_band"] == "early"
    assert summary["content_preservation_mean"] > 0.5
    print("PASS")


if __name__ == "__main__":
    main()
