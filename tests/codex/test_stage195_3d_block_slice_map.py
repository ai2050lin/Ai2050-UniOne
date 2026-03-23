#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage195_3d_block_slice_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["block_count"] == 5
    assert summary["strongest_block_name"] == "差分-路径切面"
    assert summary["weakest_block_name"] == "回收闭合切面"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
