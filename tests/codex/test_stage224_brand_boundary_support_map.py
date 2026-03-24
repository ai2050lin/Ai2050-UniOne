#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage224_brand_boundary_support_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["weakest_piece_name"] == "品牌边界就绪度"
    assert summary["strongest_piece_name"] == "水果义边界对照"
    assert summary["top_gap_name"] == "品牌边界未站住"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
