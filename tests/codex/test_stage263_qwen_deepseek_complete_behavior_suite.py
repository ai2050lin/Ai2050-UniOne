#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage263_qwen_deepseek_complete_behavior_suite import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=False)
    assert summary["model_count"] == 2
    assert summary["probe_count_per_model"] >= 10
    for row in summary["model_rows"]:
        assert row["direct_score"] >= 0.4
    print("PASS")


if __name__ == "__main__":
    main()

