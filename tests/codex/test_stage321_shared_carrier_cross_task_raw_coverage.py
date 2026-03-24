#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage321_shared_carrier_cross_task_raw_coverage import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage321_shared_carrier_cross_task_raw_coverage"
    assert 0.0 <= summary["raw_cross_task_score"] <= 1.0
    assert len(summary["task_rows"]) == 2
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
