#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage266_qwen_deepseek_parameter_hook_compare import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert len(summary["model_rows"]) == 2
    for row in summary["model_rows"]:
        assert row["parameter_hook_score"] > 0.0
        assert row["contrast_count"] >= 3
    print("PASS")


if __name__ == "__main__":
    main()

