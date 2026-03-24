#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage253_deepseek14b_translation_behavior_probe import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["translate_english_phrase"] != ""
    assert summary["translate_ascii_ratio"] >= summary["plain_ascii_ratio"]
    print("PASS")


if __name__ == "__main__":
    main()
