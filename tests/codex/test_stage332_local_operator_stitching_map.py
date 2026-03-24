#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage332_local_operator_stitching_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage332_local_operator_stitching_map"
    assert 0.0 <= summary["stitching_score"] <= 2.0
    assert len(summary["stitch_rows"]) == 4
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
