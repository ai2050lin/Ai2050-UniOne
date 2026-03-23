#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage149_rolling_expansion_scheduler import run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["status_short"] == "rolling_expansion_scheduler_ready"
    assert summary["variable_count"] == 6
    assert summary["highest_priority_variable"] in {"a", "b", "g", "r", "f", "q"}
    assert summary["lowest_priority_variable"] in {"a", "b", "g", "r", "f", "q"}
    assert len(summary["schedule_rows"]) == 6
    assert summary["total_additional_case_count"] > 0
    assert Path("tests/codex_temp/stage149_rolling_expansion_scheduler_20260323/summary.json").exists()
    assert Path("tests/codex_temp/stage149_rolling_expansion_scheduler_20260323/STAGE149_ROLLING_EXPANSION_SCHEDULER_REPORT.md").exists()
    print("PASS")


if __name__ == "__main__":
    main()
