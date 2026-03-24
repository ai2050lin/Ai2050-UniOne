#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage213_timing_phase_route_bridge import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["weakest_piece_name"] == "时序痕迹桥"
    assert summary["strongest_piece_name"] == "路径分裂桥"
    assert summary["dominant_band_name"] == "early"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
