#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage244_deepseek14b_stress_long_chain_probe import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["correct_count"] >= 5
    assert summary["stress_score"] >= 0.625
    print("PASS")


if __name__ == "__main__":
    main()
