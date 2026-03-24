#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage216_early_timing_phase_split_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["dominant_band_name"] == "early"
    assert summary["weakest_piece_name"] == "时序触发"
    assert summary["strongest_piece_name"] == "早层路径分流"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
