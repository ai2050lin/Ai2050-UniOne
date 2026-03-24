#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage227_brand_trigger_word_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["trigger_count"] == 5
    assert summary["best_trigger_word"] == "iphone"
    assert summary["worst_trigger_word"] == "store"
    assert summary["top_gap_name"] == "品牌边界触发不足"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
