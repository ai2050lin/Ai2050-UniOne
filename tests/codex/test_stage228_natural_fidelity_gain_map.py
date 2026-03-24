#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage228_natural_fidelity_gain_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["best_gain_piece_name"] == "理论提升空间"
    assert summary["worst_gain_piece_name"] == "当前天然保真"
    assert summary["top_gap_name"] == "天然来源保真不足"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
