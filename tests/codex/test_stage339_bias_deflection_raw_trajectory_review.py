#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage339_bias_deflection_raw_trajectory_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage339_bias_deflection_raw_trajectory_review"
    assert 0.0 <= summary["review_score"] <= 2.0
    assert len(summary["trajectory_rows"]) == 3
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
