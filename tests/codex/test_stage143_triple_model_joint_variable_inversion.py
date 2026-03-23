#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage143_triple_model_joint_variable_inversion import run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["status_short"] == "triple_model_joint_inversion_ready"
    assert summary["model_count"] == 3
    assert summary["family_count"] == 5
    assert summary["row_count"] == 15
    assert 0.0 <= summary["joint_inversion_score"] <= 1.0
    assert Path("tests/codex_temp/stage143_triple_model_joint_variable_inversion_20260323/summary.json").exists()
    assert Path("tests/codex_temp/stage143_triple_model_joint_variable_inversion_20260323/STAGE143_TRIPLE_MODEL_JOINT_VARIABLE_INVERSION_REPORT.md").exists()
    print("PASS")


if __name__ == "__main__":
    main()
