#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage219_attention_route_split_bridge import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["dominant_band_name"] == "early"
    assert summary["weakest_piece_name"] == "品牌义注意力取回"
    assert summary["strongest_piece_name"] == "动作选路"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
