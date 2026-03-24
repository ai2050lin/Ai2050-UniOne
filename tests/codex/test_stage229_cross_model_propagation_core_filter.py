#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage229_cross_model_propagation_core_filter import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["filtered_core_count"] == 3
    assert "条件场" in summary["filtered_core_names"]
    assert "副词动态选路" in summary["excluded_weak_names"]
    assert summary["top_gap_name"] == "硬主核仍偏少"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
