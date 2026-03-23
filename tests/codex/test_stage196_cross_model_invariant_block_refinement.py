#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage196_cross_model_invariant_block_refinement import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["stable_block_count"] == 2
    assert "条件场" in summary["stable_block_names"]
    assert "副词动态选路" in summary["weak_block_names"]
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
