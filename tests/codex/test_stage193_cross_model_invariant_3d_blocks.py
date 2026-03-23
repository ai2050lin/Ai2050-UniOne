#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage193_cross_model_invariant_3d_blocks import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["block_count"] == 5
    assert summary["strongest_block_name"] == "条件场"
    assert summary["weakest_block_name"] == "副词动态选路"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
