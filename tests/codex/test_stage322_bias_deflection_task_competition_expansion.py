#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage322_bias_deflection_task_competition_expansion import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage322_bias_deflection_task_competition_expansion"
    assert 0.0 <= summary["task_competition_score"] <= 1.0
    assert len(summary["task_rows"]) >= 2
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
