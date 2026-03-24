#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage233_brand_parameter_trigger_lattice import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 5
    assert summary["best_trigger_word"] == "iphone"
    assert summary["top_gap_name"] == "品牌义参数晶格仍然稀薄"
    assert summary["weakest_piece_name"] == "品牌义边界分裂就绪度"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
