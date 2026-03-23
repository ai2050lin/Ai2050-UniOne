#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage148_variable_identifiability_dataset_pack import run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["status_short"] == "variable_identifiability_dataset_ready"
    assert summary["variable_count"] == 6
    assert summary["family_count"] == 6
    assert summary["difficulty_count"] == 4
    assert summary["contrast_type_count"] == 4
    assert summary["case_count"] == 704
    assert 0.0 <= summary["overall_identifiability_score"] <= 1.0
    assert Path("tests/codex_temp/stage148_variable_identifiability_dataset_pack_20260323/summary.json").exists()
    assert Path("tests/codex_temp/stage148_variable_identifiability_dataset_pack_20260323/variable_bundle_catalog.json").exists()
    assert Path("tests/codex_temp/stage148_variable_identifiability_dataset_pack_20260323/identifiability_cases.jsonl").exists()
    print("PASS")


if __name__ == "__main__":
    main()
