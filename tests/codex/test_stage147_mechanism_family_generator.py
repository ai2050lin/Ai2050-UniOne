#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage147_mechanism_family_generator import run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["status_short"] == "mechanism_family_generator_ready"
    assert summary["family_count"] == 6
    assert summary["difficulty_count"] == 4
    assert summary["control_type_count"] == 4
    assert summary["target_variable_count"] == 6
    assert summary["case_count"] == 384
    assert 0.0 <= summary["mechanism_family_generator_score"] <= 1.0
    assert Path("tests/codex_temp/stage147_mechanism_family_generator_20260323/summary.json").exists()
    assert Path("tests/codex_temp/stage147_mechanism_family_generator_20260323/family_catalog.json").exists()
    assert Path("tests/codex_temp/stage147_mechanism_family_generator_20260323/cases.jsonl").exists()
    assert Path("tests/codex_temp/stage147_mechanism_family_generator_20260323/cases.csv").exists()
    print("PASS")


if __name__ == "__main__":
    main()
