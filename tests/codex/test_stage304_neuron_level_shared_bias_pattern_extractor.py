#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage304_neuron_level_shared_bias_pattern_extractor import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["extraction_score"] > 0.0
    assert summary["shared_core_count"] > 0
    assert summary["bias_core_count"] > 0
    print("PASS")


if __name__ == "__main__":
    main()
