#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage326_task_bias_raw_competition_thickening import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage326_task_bias_raw_competition_thickening"
    assert 0.0 <= summary["thickening_score"] <= 1.0
    assert len(summary["thickened_rows"]) >= 2
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
