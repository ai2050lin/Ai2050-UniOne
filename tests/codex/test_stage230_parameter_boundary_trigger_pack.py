#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage230_parameter_boundary_trigger_pack import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["top_gap_name"] == "品牌义参数边界仍未站住"
    assert summary["weakest_piece_name"] == "参数边界分裂就绪度"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
