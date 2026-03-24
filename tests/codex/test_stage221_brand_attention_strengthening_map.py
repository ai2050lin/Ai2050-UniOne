#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage221_brand_attention_strengthening_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["top_gap_name"] == "品牌义取回不足"
    assert summary["weakest_piece_name"] == "品牌边界冲突反向值"
    assert summary["strongest_piece_name"] == "水果义取回对照"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
