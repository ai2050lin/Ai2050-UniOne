#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage223_natural_vs_repair_closure_split_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["weakest_piece_name"] == "跨模型传播不变量"
    assert summary["strongest_piece_name"] == "修复闭合"
    assert summary["top_gap_name"] == "跨模型传播不变量偏少"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
