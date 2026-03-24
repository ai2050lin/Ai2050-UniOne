#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage241_deepseek14b_long_chain_probe import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["correct_count"] >= 4
    assert summary["long_chain_score"] >= 0.66
    print("PASS")


if __name__ == "__main__":
    main()
