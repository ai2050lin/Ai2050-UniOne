#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage265_qwen_deepseek_complete_final_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=False)
    assert len(summary["model_rows"]) == 2
    assert summary["score_gap"] >= 0.0
    print("PASS")


if __name__ == "__main__":
    main()
