#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage150_three_model_language_kernel_evaluation import run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["status_short"] == "three_model_language_kernel_ready"
    assert summary["model_count"] == 3
    assert summary["phenomenon_count"] == 7
    assert summary["stable_core_count"] + summary["partial_core_count"] + summary["weak_core_count"] == 7
    assert 0.0 <= summary["overall_kernel_score"] <= 1.0
    assert len(summary["model_rows"]) == 3
    assert len(summary["phenomenon_rows"]) == 7
    assert Path("tests/codex_temp/stage150_three_model_language_kernel_evaluation_20260323/summary.json").exists()
    assert Path("tests/codex_temp/stage150_three_model_language_kernel_evaluation_20260323/STAGE150_THREE_MODEL_LANGUAGE_KERNEL_EVALUATION_REPORT.md").exists()
    print("PASS")


if __name__ == "__main__":
    main()
