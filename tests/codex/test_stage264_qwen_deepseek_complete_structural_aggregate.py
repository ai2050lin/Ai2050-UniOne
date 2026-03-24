#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage264_qwen_deepseek_complete_structural_aggregate import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=False)
    assert len(summary["model_rows"]) == 2
    for row in summary["model_rows"]:
        assert row["historical_structure_score"] > 0.2
        assert row["complete_score"] > 0.3
    print("PASS")


if __name__ == "__main__":
    main()

