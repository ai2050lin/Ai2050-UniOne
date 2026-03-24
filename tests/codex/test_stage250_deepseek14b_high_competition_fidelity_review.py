#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage250_deepseek14b_high_competition_fidelity_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["probe_count"] >= 8
    assert summary["correct_count"] >= 4
    print("PASS")


if __name__ == "__main__":
    main()
