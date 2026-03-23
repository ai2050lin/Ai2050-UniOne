#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage204_phase_route_split import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["dominant_band_name"] == "early"
    assert summary["weakest_band_name"] == "late"
    assert summary["route_split_margin"] > 0
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
