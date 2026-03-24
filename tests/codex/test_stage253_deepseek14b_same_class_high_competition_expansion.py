#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage253_deepseek14b_same_class_high_competition_expansion import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["probe_count"] >= 11
    assert summary["correct_count"] >= 4
    assert summary["brand_score"] >= summary["fruit_score"]
    print("PASS")


if __name__ == "__main__":
    main()
