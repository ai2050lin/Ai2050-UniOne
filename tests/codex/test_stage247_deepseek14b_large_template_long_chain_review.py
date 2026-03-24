#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage247_deepseek14b_large_template_long_chain_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["correct_count"] >= 8
    assert summary["review_score"] >= 0.6667
    print("PASS")


if __name__ == "__main__":
    main()
