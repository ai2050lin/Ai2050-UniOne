#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage306_neuron_level_math_principle_summary import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["principle_score"] > 0.0
    assert len(summary["operator_triplet"]) == 3
    print("PASS")


if __name__ == "__main__":
    main()
