#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage142_cross_model_anaphora_hardening import run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["status_short"] == "cross_model_anaphora_hardening_ready"
    assert summary["model_count"] == 3
    assert summary["family_count"] == 5
    assert 0.0 <= summary["mean_model_hardening_score"] <= 1.0
    assert 0.0 <= summary["all_model_positive_late_rescue_family_rate"] <= 1.0
    assert Path("tests/codex_temp/stage142_cross_model_anaphora_hardening_20260323/summary.json").exists()
    assert Path("tests/codex_temp/stage142_cross_model_anaphora_hardening_20260323/STAGE142_CROSS_MODEL_ANAPHORA_HARDENING_REPORT.md").exists()
    print("PASS")


if __name__ == "__main__":
    main()
