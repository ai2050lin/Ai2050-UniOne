#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage256_deepseek14b_multidirection_translation_probe import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["probe_count"] >= 4
    assert summary["behavior_score"] >= 0.75
    print("PASS")


if __name__ == "__main__":
    main()
