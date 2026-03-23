#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage207_phase_timing_coupling import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["strongest_piece_name"] == "路径分裂"
    assert summary["weakest_piece_name"] == "时序痕迹"
    assert summary["coupling_score"] > 0.5
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
