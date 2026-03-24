#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage267_qwen_deepseek_same_class_competition_compare import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert len(summary["model_rows"]) == 2
    for row in summary["model_rows"]:
        assert row["probe_count"] >= 5
    print("PASS")


if __name__ == "__main__":
    main()

